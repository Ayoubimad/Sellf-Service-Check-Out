import supervision as sv
import torch


from faster_rcnn import Faster_RCNN
from utils import load_image, annotate, get_random_image_path

if __name__ == "__main__":

    detector = Faster_RCNN()

    detector.load_weights("./weights/faster_rcnn.pth")
    image_dir = "datasets/UNIBIM2016/original"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _ in range(5):
        image_path = get_random_image_path(image_dir)

        _, img = load_image(image_path)

        # per una sola immagine
        predictions = detector.predict(img)
        # oppure per un batch di immagini

        # predictions = detector([img.to(DEVICE) for img in images])
        # images è un tensore di shape (batch_size, 3, 224, 224)

        img, _ = load_image(image_path)  # BGR

        img = annotate(
            image_source=img,
            boxes=predictions["boxes"].numpy(),
            confidence=predictions["scores"].numpy(),
            class_id=predictions["labels"].numpy(),
        )

        sv.plot_image(img, (8, 8))
