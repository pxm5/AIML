import torch
import torch.nn as nn
import torch.nn.functional as Fun
from tqdm import tqdm
import timm


class Classifier(nn.Module):

    def __init__(self, classes=2) -> None:
        super(Classifier, self).__init__()
        self.base_model = timm.create_model('resnet50', pretrained=False)
        self.linear = nn.Linear(in_features=2048, out_features=classes, bias=True)
        self.model = nn.Sequential(*list(self.base_model.children())[:-1], self.linear)

    def forward(self, X):
        out = self.model(X)
        return out

       
def trainModel(model:Classifier, epochs:int, opti, crit,  train_loader, valid_loader=None, validate=False):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	# train_loader.to(device)
	model.train()
	running_loss = 0.
	for i in range(epochs):
		for images, labels in tqdm(train_loader, desc= f'Training {i + 1}/{epochs} epoch'):
			opti.zero_grad()
			output = model(images).to(device)
			labels = labels.to(device)
			loss = crit(output, labels)
			loss.backward()
			opti.step()
			running_loss += loss.item() / labels.size(0)
			correct_preds = torch.sum(torch.argmax(output, dim=1) == labels)
		print(f'running loss: {running_loss} epoch; {i + 1}, ==={(correct_preds/labels.size(0)) * 100}%===')

		if validate:
			# valid_loader.to(device)
			model.eval()
			running_loss = 0.
			with torch.no_grad():
				for images, labels in tqdm(valid_loader, desc= f'Validation {i + 1}/{epochs} epoch'):
					output = model(images).to(device)
					labels = labels.to(device)
					loss = crit(output, labels)
					running_loss += loss.item() / labels.size(0)
					correct_preds = torch.sum(torch.argmax(output, dim=1) == labels)
					print(f'running loss: {running_loss} epoch; {i + 1},  ==={(correct_preds/labels.size(0)) * 100}%===')

	print('TRAINING COMPLETE')
	
def testModel(model:Classifier, crit, test_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # test_loader.to(device)
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f'Testing:'):
            output = model(images).to(device)
            labels = labels.to(device)
            loss = crit(output, labels)
            correct_preds = torch.sum(torch.argmax(output, dim=1) == labels)
            running_loss += loss.item() / labels.size(0)

        print(f'Running Loss: {running_loss} \n Average Loss Per Batch: {running_loss/test_loader.__len__()}; ==={(correct_preds/labels.size(0))*100}%')
