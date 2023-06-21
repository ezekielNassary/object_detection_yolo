import cv2
import numpy as np
class ObjectDetection:
    def __init__(self, image_path, config_path, weights_path, classes_path):
        self.image_path = image_path
        self.config_path = config_path
        self.weights_path = weights_path
        self.classes_path = classes_path
    def pixels_to_inches(pixels, dpi):
        inches = pixels / dpi
        cm=inches*2.54
        return cm
    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = self.COLORS[class_id]
        #cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        #cv2.rectangle(img, (x, y), (x + x_plus_w, y + y_plus_h), color,2)
        print(label)
        #cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def run_detection(self):
        image = cv2.imread(self.image_path)
        print("processing....")
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        self.classes = None
        with open(self.classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

        net = cv2.dnn.readNet(self.weights_path, self.config_path)

        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(self.get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    dpi = 40
                    physical_width = pixels_to_inches(w, dpi)
                    physical_height = pixels_to_inches(h, dpi)
                    
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    print(physical_height)
                    

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            try:
                box = boxes[i]
            except:
                i = i[0]
                box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        #cv2.imshow("object detection", image)
        #cv2.waitKey(1)
        cv2.imwrite("object-detection.jpg", image)
        #cv2.destroyAllWindows()
        print("data sent")

