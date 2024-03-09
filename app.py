import torch
from train_models.train import TransferNet
import requests
from PIL import Image
from torchvision import transforms
import gradio as gr
from train_models.train import run
device = torch.device('cpu')
model = TransferNet()
model.load_state_dict(torch.load("../../hp/Eye-Retina-Degeneration-Detection/models/transfer_model.pth", map_location=device))

labels = ["DR", "No_DR"]

def predict(image_path):
    
    print(image_path)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    predicted_class_label = "Diabetic Retinopathy Detected ⚠️" if predicted_class == 0 else "No Diabetic Rhetinopathy Detected ✅"

    probability = probabilities[0][predicted_class].cpu().numpy() * 100
    rounded_probability = round(probability, 2)
    rounded_probability = str(rounded_probability) + "%"
    return predicted_class_label, rounded_probability
    
gradient_input = [
    gr.Slider(minimum=1, maximum=500, step = 10, label="Number of Epoches"),
    gr.Radio(label="Optimizers", choices = [("Adam", 0), ("RMSprop", 1), ("Adagrad", 2), ("SGD", 3)]),
    gr.Slider(minimum=0.000001, maximum=1, step = 0.001, label="Learning Rate"),
    ]

gradient_output = [
    gr.Textbox(label="Accuracy Score"),
    gr.Image(label = "Confusion Matrix"),
    gr.Image(label = "Loss Graph")
]

inp = [
        gr.Image(type="filepath") 
    ]

output =  [ gr.Label(label="Predction: "),
            gr.Label(label="Probability: ")
        ]  

one = gr.Interface(
    fn = run,
    inputs = gradient_input,
    outputs = gradient_output,  
    submit_btn = "Train",
    title="Train your own model!",
    description="<img src='https://i.ibb.co/Bw08434/logo-1.png' alt='Logo' style='width:230px;height:100px;border-radius:5px;box-shadow:2px 2px 5px 0px rgba(0,0,0,0.75);background-color:black;'><br>",
    article="<h3>Dataset link here: <a href='https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy'>Dataset</a>.</h3>"
    
    
    
)
two = gr.Interface(
    fn = predict,
    inputs = inp,
    outputs = output, 
    submit_btn="Predict",
    title="Predict Diabetic Retinopathy 👁!",
    description="<img src='https://i.ibb.co/Bw08434/logo-1.png' alt='Logo' style='width:230px;height:100px;border-radius:5px;box-shadow:2px 2px 5px 0px rgba(0,0,0,0.75);background-color:black;'><br>",
    examples=
    [
        ["../../hp/Eye-Retina-Degeneration-Detection/dataset/test/DR/0ada12c0e78f_png.rf.3e8e491a2cacb9af201e2f89f3afca61.jpg", "DR"],
        ["../../hp/Eye-Retina-Degeneration-Detection/dataset/test/DR/2fdffb6160a6_png.rf.fde20941f556703f7b478d53e28950d1.jpg", "DR"],
        ["../../hp/Eye-Retina-Degeneration-Detection/dataset/test/No_DR/0ae2dd2e09ea_png.rf.a4faf61bd46dc2930c51b3db7dba12cd.jpg", "No_DR"],
        ["../../hp/Eye-Retina-Degeneration-Detection/dataset/test/No_DR/03b373718013_png.rf.aeac7af7a221106fab6aaa133b5ecc3f.jpg", "No_DR"],
        
    ]
    
    )

demo = gr.TabbedInterface([one, two], ["Train", "Predict"])

if __name__ == "__main__":
    demo.launch()