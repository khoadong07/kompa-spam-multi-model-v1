from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from litserve import LitAPI
import os

LOCAL_MODEL_DIR = "./models"

CATEGORY_MODEL_MAP = {
    "healthcare_insurance": {
        "repo_id": "Khoa/kompa-spam-filter-healthcare-insurance-update-0525",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-healthcare-insurance-update-0525")
    },
    "energy_fuels": {
        "repo_id": "Khoa/kompa-spam-filter-energy-fuels-update-0625",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-energy-fuels-update-0625")
    },
    "electronic": {
        "repo_id": "Khoa/kompa-spam-filter-electronic-update-0625",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-electronic-update-0625")
    },
    "fmcg": {
        "repo_id": "Khoa/kompa-spam-filter-fmcg-update-0625",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-fmcg-update-0625")
    },
    "fnb": {
        "repo_id": "Khoa/kompa-spam-filter-fnb-update-0625",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-fnb-update-0625")
    },
    "logistic_delivery": {
        "repo_id": "Khoa/kompa-spam-filter-logistics-delivery-update-0625",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-logistics-delivery-update-0625")
    },
    "bank": {
        "repo_id": "Khoa/kompa-spam-filter-bank-update-0625",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-bank-update-0625")
    },
    "finance": {
        "repo_id": "Khoa/kompa-spam-filter-finance-update-0525",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-finance-update-0525")
    },
    "ewallet": {
        "repo_id": "Khoa/kompa-spam-filter-e-wallet-update-0625",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-e-wallet-update-0625")
    },
    "investment": {
        "repo_id": "Khoa/kompa-spam-filter-investment-update-0625",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-investment-update-0625")
    },
    "real_estate": {
        "repo_id": "Khoa/kompa-spam-filter-real-estate-update-0525",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-real-estate-update-0525")
    },
    "education": {
        "repo_id": "Khoa/kompa-spam-filter-education-update-0625",
        "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-education-update-0625")
    },
}

from huggingface_hub import snapshot_download

class SpamLitAPI(LitAPI):
    def setup(self, device):
        self.models = {}
        self.device = device

        for category, info in CATEGORY_MODEL_MAP.items():
            model_path = info["local_path"]
            if not os.path.exists(model_path):
                print(f"📥 Downloading model for {category} from {info['repo_id']}")
                snapshot_download(repo_id=info["repo_id"], local_dir=model_path, local_dir_use_symlinks=False)

            print(f"✅ Loading model for {category} from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            model.to(device)
            model.eval()

            self.models[category] = {
                "model": model,
                "tokenizer": tokenizer
            }

    def decode_request(self, request):
        text = request["text"]
        category = request.get("category")
        if not category or category not in self.models:
            raise ValueError(f"Invalid or unsupported category: {category}")

        tokenizer = self.models[category]["tokenizer"]
        if isinstance(text, str):
            text = [text]
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=250)
        return {"inputs": inputs, "category": category}

    def predict(self, batch):
        inputs = batch["inputs"]
        category = batch["category"]
        model = self.models[category]["model"]
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = model(**inputs).logits
        return logits

    def encode_response(self, logits):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        results = []
        for i in range(probs.size(0)):
            spam_score = probs[i][1].item()
            is_spam = spam_score > 0.5
            results.append({
                "spam": is_spam,
                "spam_score": round(spam_score, 4),
                "all_probs": {
                    "not_spam": round(probs[i][0].item(), 4),
                    "spam": round(probs[i][1].item(), 4)
                }
            })
        return results if len(results) > 1 else results[0]

from litserve import LitServer

if __name__ == "__main__":
    api = SpamLitAPI()
    server = LitServer(api, accelerator="cpu", devices=0) 
    server.run(host="0.0.0.0", port=8989, num_api_servers=8, log_level="info")
