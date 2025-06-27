from transformers import AutoModelForObjectDetection
import torchvision.transforms as T
from extract_from_pdf import MaxResize, pdf2images, show_images
import torch
from PIL import Image
import os

detection_transformer = T.Compose(
    [
        MaxResize(1600),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)
model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return boxes

def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score), 'bbox': [float(elem) for elem in bbox]})

    return objects

def detect_and_crop_save_table(file_path):
    img = Image.open(file_path)
    print(img.size)
    filename, _ = os.path.splitext(os.path.basename(file_path))
    cropped_table_dir = os.path.join(os.path.dirname(file_path), 'table_images')

    if not os.path.exists(cropped_table_dir):
        os.makedirs(cropped_table_dir)
    
    pixel_values = detection_transformer(img).unsqueeze(0)
    print(pixel_values.shape)
    with torch.no_grad():
        outputs = model(pixel_values)

    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    detected_tables = outputs_to_objects(outputs, img.size, id2label)

    print(f"Detected {len(detected_tables)} table(s)")

    for idx in range(len(detected_tables)):
        cropped_table = img.crop(detected_tables[idx]['bbox'])
        cropped_table.save(os.path.join(cropped_table_dir, f"{filename}_table_{idx}.png"))


if __name__ == "__main__":
    detect_and_crop_save_table("Diagnosis/page_5.png")
    show_images("Diagnosis/table_images")
