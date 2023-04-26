from util.makeRandomQR import randomQR
from util.qrCodeValidator import qrValidator
from util.do_affineImage import do_affine
from PIL import Image
from tqdm import tqdm
from pyzbar import pyzbar

def get_good_affine_qr()->Image.Image:
    affined = None
    i = 0
    pbar = tqdm(range(10))
    complexity = 100
    while True:
        qr, rand_data, v = randomQR.get_random_qrcode(complexity=complexity)
        affined = do_affine.affine_transform_v2(qr)
        if qrValidator.is_qrcode(affined):
            # doubel decode is pass
            if pyzbar.decode(qr)[0].data.decode('utf-8') == pyzbar.decode(affined)[0].data.decode('utf-8'):
                break
        i += 1
        complexity -= 2
        pbar.set_description("re-gen times :{:>5}".format(i))
    good = qr
    affined = Image.fromarray(affined)
    return good, affined

if __name__ == "__main__":
    n = 100
    pbar = tqdm(range(n))
    for i, _ in enumerate(pbar):
        pbar.set_description(f"{i+1}/{n}")
        good, affined = get_good_affine_qr()
        good.save(f"./data/good/{i}.png")
        affined.save(f"./data/affine/{i}.png")