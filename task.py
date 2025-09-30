import os
import time
import json
from google.cloud import pubsub_v1
from torch.utils.tensorboard import SummaryWriter  # se usar PyTorch

PROJECT = os.environ["GCP_PROJECT"]
TOPIC = os.environ["TRAINING_PUBSUB_TOPIC"]
JOB_ID = os.environ.get("JOB_ID", "job-unknown")

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT, TOPIC)

# Tensorboard log dir (Vertex sets AIP_TENSORBOARD_LOG_DIR)
tb_logdir = os.environ.get("AIP_TENSORBOARD_LOG_DIR", "/tmp/tb")
writer = SummaryWriter(tb_logdir)

def publish_update(step, loss, attention_uri=None):
    payload = {
        "job_id": JOB_ID,
        "step": step,
        "loss": float(loss),
        "attention_uri": attention_uri,
        "timestamp": time.time()
    }
    publisher.publish(topic_path, json.dumps(payload).encode("utf-8"))

def train():
    for step in range(1000):
        loss = 1.0 / (step+1)  # dummy
        writer.add_scalar("loss", loss, step)
        if step % 10 == 0:
            # opcional: salve snapshot de attention em GCS e envie URI
            attention_uri = f"gs://my-bucket/{JOB_ID}/attention/step_{step}.npz"
            publish_update(step, loss, attention_uri)
        time.sleep(0.1)

if __name__ == "__main__":
    train()
    writer.close()