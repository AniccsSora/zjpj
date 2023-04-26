import qrcode
import random
from PIL import Image

def get_random_qrcode(complexity=100)->(Image.Image, str):
    k = random.randint(1, 999)
    data = ''.join(random.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz', k=k))

    _ = 8 if len(data)*(complexity/100) < 8 else int(len(data)*(complexity/100))
    data = data[0:_]

    base_version=40
    # Generate a random version number for the QR code
    _ = 2 if base_version * (complexity/100) < 2 else int(base_version * (complexity/100))
    version = random.randint(1, _)

    # Create a QR code with the data and version number
    ec_l = [qrcode.constants.ERROR_CORRECT_L,
            qrcode.constants.ERROR_CORRECT_M,
            qrcode.constants.ERROR_CORRECT_Q,
            qrcode.constants.ERROR_CORRECT_H ]
    ec = random.choice(ec_l)
    # print("error correct:", ec)
    qr = qrcode.QRCode(version=version, error_correction=ec)
    qr.add_data(data)
    qr.make(fit=True)

    # Create an image from the QR code
    img = qr.make_image(fill_color="black", back_color="white")
    # print("version:", version)
    # print("size:", img.size)
    if len(img.getbands()) == 1:
        img = img.convert("RGB")
    # print("channels:", len(img.getbands()))
    return img, data, version

if __name__ == "__main__":

    img = get_random_qrcode()
