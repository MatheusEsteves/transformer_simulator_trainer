import os
import time
import json
from google.cloud import pubsub_v1
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from transformers import BertTokenizer

# Constantes para parametrizações de configuração no PubSub
PROJECT = "SIMPLE_TRANSFORMER_GCP_PROJECT"  # TODO os.environ["GCP_PROJECT"]
TOPIC = "SIMPLE_TRANSFORMER_TRAINING_PUBSUB_TOPIC"  # TODO os.environ["TRAINING_PUBSUB_TOPIC"]
JOB_ID = "SIMPLE_TRANSFORMER_JOB"  # TODO os.environ.get("JOB_ID", "job-unknown")

# Constantes para parametrizações do modelo Transformer Simples
VOCAB_SIZE = 30522  # Usando o vocabulário do BERT
EMBED_SIZE = 3
NUM_HEADS = 1
NUM_STEPS = 100
NUM_PUBLISHER_BLOCKS = 10

# Definindo o modelo Transformer simples
class SimpleTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads):
        super(SimpleTransformerModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size)  # Embedding de 3 dimensões
        self.attention = nn.MultiheadAttention(embed_size, num_heads)  # Camada de atenção multi-cabeça
        self.fc = nn.Linear(embed_size, vocab_size)  # Camada de saída para predição de palavras

        self.d_k = int(embed_size/num_heads)
        self.n_heads = num_heads

        # matrizes WQ, WK e WV que criaram as matrizes de
        # vetores Q, K e V para nossas cabeças de atenção
        self.Wq = nn.Linear(embed_size, embed_size)
        self.Wk = nn.Linear(embed_size, embed_size)
        self.Wv = nn.Linear(embed_size, embed_size)
        
    def forward(self, x):
        # Aplica o embedding
        x = self.embedding(x)
        
        # Transpor para formato adequado para a camada de atenção
        x = x.transpose(0, 1)  # A camada MultiheadAttention espera [seq_len, batch_size, embed_size]

        # Gera as matrizes de vetores Q, K e V
        Q = self.Wq(x)  # [seq_len, batch_size, embed_size]
        K = self.Wk(x)  # [seq_len, batch_size, embed_size]
        V = self.Wv(x)  # [seq_len, batch_size, embed_size]

        # Passando Q, K, V pela camada de atenção
        output, attention_weights = self.attention(Q, K, V)  # Aqui as entradas são as mesmas, mas isso pode ser alterado
        attention_weights = torch.tensor([[attention_weights.tolist()]])
        attention_output = output

        # A camada de saída
        output = self.fc(attention_output)
        return output, attention_weights

# Definindo classe para operações de treinamento usando o modelo de Transformer simples
class SimpleTransformerTrainer():
    def __init__(self, model, text_sentence_train, text_sentence_train_b):
        super(SimpleTransformerTrainer, self).__init__()

        self.model = model
        self.text_sentence_train = text_sentence_train
        self.text_sentence_train_b = text_sentence_train_b

        # Inicializando o tokenizador
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize(self):
        # Tokenizando o texto
        inputs = self.tokenizer(self.text_sentence_train, self.text_sentence_train_b, return_tensors='pt')
        self.input_ids = inputs['input_ids']  # IDs dos tokens

        # Apenas para uso da renderização dos pesos e tokens de atenção no BertViz
        # ------------------------------------------------------------------------
        # input_id_list = input_ids[0].tolist()
        # token_type_ids = inputs['token_type_ids']
        # tokens = self.tokenizer.convert_ids_to_tokens(input_id_list)
        # sentence_b_start = token_type_ids[0].tolist().index(1)

        # Definindo a função de perda e o otimizador
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        
        output, attention_weights = self.model(self.input_ids)  # Passa os dados pelo modelo

        output = output.transpose(0, 1)  # Reverter a transposição
        output = output.view(-1, self.model.vocab_size)  # Redimensionar para a forma correta
        self.input_ids = self.input_ids.view(-1)  # Redimensionar para a forma correta
        loss = self.criterion(output, self.input_ids)  # Calcular a perda

        loss.backward()  # Retropropagar
        self.optimizer.step()  # Atualizar os pesos
        return loss.item(), attention_weights
    
class PubSubPublisherClient():
    def __init__(self, job_id, project, topic):
        super(SimpleTransformerTrainerManager, self).__init__()

        # Definindo configurações para conexão com o pubsub client
        self.job_id = job_id
        self.project = project
        self.topic = topic
        
        self.pubsub_publisher_client = pubsub_v1.PublisherClient()
        self.topic_path = self.pubsub_publisher_client.topic_path(project, topic)

        # Tensorboard
        self.tb_logdir = os.environ.get("AIP_TENSORBOARD_LOG_DIR", "/tmp/tb")
        self.writer = SummaryWriter(self.tb_logdir)

    def build_payload(self, step, key_values):
        payload = {
            "job_id": self.job_id,
            "step": step,
            "timestamp": time.time()
        }

        for key in key_values:
            payload[key] = key_values[key]

        return payload

    def add_scalar(self, key, value, step):
        self.writer.add_scalar(key, value, step)

    def publish_update(self, payload):
        self.pubsub_publisher_client.publish(self.topic_path, json.dumps(payload).encode("utf-8"))

    def close(self):
        self.writer.close()

# Definindo classe para operações de treinamento conforme o Nº de iterações parametrizado pelo algoritmo, e comunicação com PubSub
class SimpleTransformerTrainerManager():
    def __init__(self, trainer, publisher, num_steps, num_publisher_blocks):
        super(SimpleTransformerTrainerManager, self).__init__()

        self.trainer = trainer
        self.publisher = publisher
        self.num_steps = num_steps
        self.num_publisher_blocks = num_publisher_blocks

    def train(self):
        for step in self.num_steps:
            loss, attention_weights = self.trainer.train()
            
            self.publisher.add_scalar("loss", loss, step)
            self.publisher.add_scalar("attention_weights", attention_weights, step)

            if step % self.num_publisher_blocks == 0:
                key_values_payload = {
                    "loss": loss,
                    "attention_weights": attention_weights
                }
                payload = self,publisher.build_payload(step, key_values_payload)

                self.publisher.publish_update(payload)

            time.sleep(0.1)

        self.publisher.close()

if __name__ == "__main__":
    model = SimpleTransformerModel(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS)

    text_sentence_train = "teste testando texto treinamento"
    text_sentence_b = "teste comparativo da outra sentenca"
    
    trainer = SimpleTransformerTrainer(model, text_sentence_train, text_sentence_b)
    publisher = PubSubPublisherClient(JOB_ID, PROJECT, TOPIC)
    trainer_manager = SimpleTransformerTrainerManager(trainer, publisher, NUM_STEPS, NUM_PUBLISHER_BLOCKS
                                                      )
    trainer_manager.train()