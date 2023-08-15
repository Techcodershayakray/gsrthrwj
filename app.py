from bardapi import Bard
from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import wikipediaapi
from bs4 import BeautifulSoup
import requests
import bardapi
import wget

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def remove_duplicates(input_array):
    return list(set(input_array))


# Download the YOLOv3 weights file from the provided URL
yolov3_weights_url = "https://github.com/Techcodershayakray/app_flask_op/blob/main/yolov3.weights"
yolov3_weights_path = "yolov3.weights"

# Download the weights file if it doesn't exist
if not os.path.exists(yolov3_weights_path):
    print("Downloading YOLOv3 weights...")
    wget.download(yolov3_weights_url, yolov3_weights_path)
    print("\nDownload complete!")

net = cv2.dnn.readNet(yolov3_weights_path, 'yolov3.cfg')
filename = ''
detected_objects = []
classes = []
classname1 = []
class_name = []
output = []
array = []
name = []
output = []
array = []
topics = []
empty_set = set()

with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


# Wikipedia API setup
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent="app/1.0"  # Replace with your app's name and version
)


@app.route('/test_again')
def test_again():
    return render_template('index.html')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        name.clear()
        detected_objects.clear()
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Object Detection
        img = cv2.imread(filename)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                label = str(classes[class_ids[i]])
                name.append(label)
                class_id = class_ids[i]
                class_name = classes[class_id]
                wiki_page = wiki_wiki.page(class_name)

                detected_object = {
                    'class': class_name,
                    'confidence': confidences[i],
                    'box': boxes[i],
                    'summary': wiki_page.summary if wiki_page.exists() else "No Wikipedia data available"
                }
                detected_objects.append(detected_object)
            filename = filename.split('_')[-1]

        return redirect(url_for('uploaded_file', filename='result_' + file.filename))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    array.clear()
    output.clear()
    topics.clear()
    filename = filename.split('_')[-1]
    n = remove_duplicates(name)

    str = ""
    for i in range(len(n) - 1):
        str = str + n[i] + ","
    str = "relationship between " + str + "and " + n[len(n) - 1] + "(only theory no venn diagram or any diagram)"
    for i in range(len(n)):
        topics.append(n[i])
    topics.append(str)
    os.environ['_BARD_API_KEY'] = "ZwgglKJi3weCmy8ZK1frwxpFoQ7v5KGW3wFiX_oE8kO5vRVYwNdVxdozI4089ssHoskP5g."

    for i in topics:
        bard_output = Bard().get_answer(i)['content']
        array.append(i)
        output.append(bard_output)

        print()
    return render_template('uploaded.html', filename=filename, name=output, array=array,
                           detected_objects=detected_objects)


@app.route('/clear_results')
def clear_results():
    return render_template('uploaded.html', filename=filename, name=[], array=[], detected_objects=[])


if __name__ == '__main__':
    app.run(debug=True)
