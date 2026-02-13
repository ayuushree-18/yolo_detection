import os
import xml.etree.ElementTree as ET

def load_xml_gt(image_name):
    """
    Loads Pascal VOC XML GT for a given image name
    """

    base = os.path.splitext(image_name)[0]
    xml_path = os.path.join("datasets", "annotations", f"{base}.xml")

    if not os.path.exists(xml_path):
        return None

    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt_boxes = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        bnd = obj.find("bndbox")

        x1 = int(bnd.find("xmin").text)
        y1 = int(bnd.find("ymin").text)
        x2 = int(bnd.find("xmax").text)
        y2 = int(bnd.find("ymax").text)

        gt_boxes.append((cls_name, x1, y1, x2, y2))

    return gt_boxes
