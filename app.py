from fastapi import FastAPI, File
from segmentation import get_yolov5, get_image_from_bytes
import uvicorn
import json
from fastapi.middleware.cors import CORSMiddleware

model = get_yolov5()

app = FastAPI(
    title="This is a YOLOV5 Cocacola-Pepsi Detection API",
    description="""Obtain object value out of image
                    and return json result""",
    version="0.0.1",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def get_health():
    return dict(msg='Cocacola and Pepsi Bottle Detection API is Ready to use.')


@app.post("/detect")
async def detect_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")
    detect_res = json.loads(detect_res)
    val = []
    res = []
    if not detect_res:
        res.append('Please send a good quality image, The model is build with a few amount of data')
    else:
        for i in detect_res:
            val.append(i['name'])
    for i in val:
        if i == 'Coca-0':
            res.append({'Brand': 'Cocacola', 'Percentage of fluid': "0%", 'amount of fluid': '0ml', 'Distinguish':
                'Cocacola'})
        if i == 'Coca-25':
            res.append({'Brand': 'Cocacola', 'Percentage of fluid': "25%", 'amount of fluid': '250ml', 'Distinguish':
                'Cocacola'})
        if i == 'Coca-50':
            res.append({'Brand': 'Cocacola', 'Percentage of fluid': "50%", 'amount of fluid': '500ml', 'Distinguish':
                'Cocacola'})
        if i == 'Coca-100':
            res.append({'Brand': 'Cocacola', 'Percentage of fluid': "90%", 'amount of fluid': '1Lt', 'Distinguish':
                'Cocacola'})

        if i == 'Pepsi-0':
            res.append({'Brand': 'Pepsi', 'Percentage of fluid': "0%", 'amount of fluid': '0ml', 'Distinguish':
                'Pepsi'})
        if i == 'Pepsi-25':
            res.append({'Brand': 'Pepsi', 'Percentage of fluid': "25%", 'amount of fluid': '250ml', 'Distinguish':
                'Pepsi'})
        if i == 'Pepsi-50':
            res.append({'Brand': 'Pepsi', 'Percentage of fluid': "50%", 'amount of fluid': '500ml', 'Distinguish':
                'Pepsi'})
        if i == 'Pepsi-75':
            res.append({'Brand': 'Pepsi', 'Percentage of fluid': "75%", 'amount of fluid': '750', 'Distinguish':
                'Pepsi'})
        if i == 'Pepsi-100':
            res.append({'Brand': 'Pepsi', 'Percentage of fluid': "90%", 'amount of fluid': '1Lt', 'Distinguish':
                'Pepsi'})


    return {"result": res}


if __name__ == '__main__':
    uvicorn.run(app, port=8000)
