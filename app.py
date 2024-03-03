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
    predicted_class = "Diabetic Retinopathy" if predicted_class == 0 else "No Diabetic Rhetinopathy"
    return predicted_class, probabilities[0].cpu().numpy()
    


# output = [ gr.Label(label="Predction: "),
#         gr.Textbox(label="Probabilities: ")]

# gr.Interface(fn=predict,
#              inputs=gr.Image(type="filepath"),
#              outputs=output,
#              examples=["../../hp/Eye-Retina-Degeneration-Detection/dataset/test/DR/0ada12c0e78f_png.rf.3e8e491a2cacb9af201e2f89f3afca61.jpg", "../../hp/Eye-Retina-Degeneration-Detection/dataset/test/DR/0bf37ca3156a_png.rf.5fd49da65121f9fd951a208b5f085744"]
#              ).launch()

def app_interface():
    with gr.Blocks() as interface:
        gr.HTML("<img src='keensight_logo.png' alt='Company Logo'>")
        with gr.Row("Eye Retina Degeneration Detection"):
            
            with gr.Column("Model Training "):
                gr.HTML("<h2>Train your own model!</h2>")
                    
                with gr.Row(""):
                    gradient_input = [
                        gr.Slider(minimum=10, maximum=500, step = 10, label="Number of Epoches")
                    ]
                with gr.Row(""):
                    gradient_output = [
                        gr.Textbox(label="Accuracy Score"),
                    ]
                    
                    gradient_train_button = gr.Button(value="Train Model")
                    
                
            with gr.Column("Please fill the form to predict insurance cost!"):
                gr.HTML("<h2>Please upload an image to detect degeneration!</h2>")
                
                inp = [
                    gr.Image(type="filepath")
                ]
                
                output =  [ gr.Label(label="Predction: "),
                            gr.Textbox(label="Probabilities: ")]
                
                predict_button = gr.Button(value="Detect")
                
        gradient_train_button.click(run, inputs=gradient_input, outputs=gradient_output)
        predict_button.click(predict, inputs=inp, outputs=output)

    interface.launch()

if __name__ == "__main__":
    app_interface()