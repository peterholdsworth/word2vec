import torch
import dataset
import model
import tqdm

ds = dataset.W2VData(3)
dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)

cbow = model.CBOW(len(ds.tokenizer.vocab), 50)
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(cbow.parameters(), lr=0.001)

for epoch in range(10):
  total_loss = 0
  for context, target in tqdm.tqdm(dl, desc=f"Epoch {epoch+1}/10", unit="batch"):
    optimizer.zero_grad()
    log_probs = cbow(context)
    loss = loss_function(log_probs, target)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  print(f"Epoch {epoch+1}/10, Loss: {total_loss}")
  torch.save(cbow.state_dict(), f"./cbow_epoch_{epoch+1}.pt")