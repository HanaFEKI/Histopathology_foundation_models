import torch
from clip import ToyTokenizer, CLIPModel, CLIPTrainer, CLIPZeroShot

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

   
    # 1. Prepare tokenizer & vocabulary
    texts = [
        "adenocarcinoma tissue",
        "benign glandular structure",
        "invasive ductal carcinoma",
        "lymphocyte infiltration"
    ]
    tokenizer = ToyTokenizer()
    tokenizer.build_vocab(texts)

    # 2. Initialize CLIP model
    model = CLIPModel(vocab_size=len(tokenizer)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    trainer = CLIPTrainer(model, optimizer)

    # 3. Prepare dummy batch
    B = 4
    # Dummy images [B,3,224,224]
    images = torch.randn(B, 3, 224, 224).to(device)
    # Dummy tokenization
    input_ids = []
    for t in texts[:B]:
        input_ids.append(tokenizer.encode(t))
    max_len = max(len(seq) for seq in input_ids)
    ids_tensor = torch.full((B, max_len), tokenizer.pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(ids_tensor)
    for i, seq in enumerate(input_ids):
        L = len(seq)
        ids_tensor[i, :L] = torch.tensor(seq, device=device)
        attention_mask[i, :L] = 1

    # 4. Training step
    loss, logit_scale = trainer.train_step(images, ids_tensor, attention_mask)
    print(f"Train loss: {loss:.4f} | Logit scale: {logit_scale:.3f}")

    # 5. Zero-shot classification
    classes = ["adenocarcinoma", "benign", "lymphocytes"]
    prompt_templates = [
        "a histopathology image of {} tissue",
        "microscopic image showing {}",
        "this is {} in H&E staining"
    ]
    zero_shot = CLIPZeroShot(model, tokenizer)
    probs = zero_shot.classify(images[:2], classes, prompt_templates)
    print("Zero-shot class probabilities for first 2 images:\n", probs)

if __name__ == "__main__":
    main()
