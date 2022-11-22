import uvicorn
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.staticfiles import StaticFiles
import asyncio
import pandas as pd
import json
from models_for_site.model1 import main_processing
from models_for_site.modelRFR import main_processing_RFR
from models_for_site.modelVoting import main_processing_voting

templates = Jinja2Templates(directory="templates")
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"))



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse("index2.html", {"request": request})

@app.post("/postdata")
async def postdata( request: Request):
    form = await request.form()
    file = form["file"]
    print("name_column_y", form["name_column_y"])
    print(file.filename)
    if '.csv' in file.filename:
        path = file.filename
        content = await file.read()
        with open(path, "w") as f:
            f.write(content.decode("utf-8"))
        df = pd.read_csv(path)
        print(df.info())
        if int(form["checkRadio"]) == 0:
            result_modeling = main_processing(df, name_y=form["name_column_y"])
        elif int(form["checkRadio"]) == 1:
            result_modeling = main_processing_RFR(df, name_y=form["name_column_y"])
        else:
            result_modeling = main_processing_voting(df, name_y=form["name_column_y"])
        trace1 = {
            "x": list(range(result_modeling["N"])),
            "y": result_modeling["rel"],
            "type": 'scatter'
        }

        trace2 = {
            "x": list(range(result_modeling["N"])),
            "y": result_modeling["pr"],
            "type": 'scatter'
        }

        trace3 = {
            "x": list(range(result_modeling["N"])),
            "y": result_modeling["y_test"],
            "type": 'scatter'
        }


        return {  "trace1": trace1, "trace2": trace2, "trace3": trace3}
    '''
    form = await request.form()
    for k, v in form.items():
        print(k,v)
    '''
    request = None #{	"x": [1, 2, 3, 4], 	  "y": [10, 15, 13, 17]}
    return {"y" : "new Y"}

@app.get("/lefttop.htm", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("/lefttop.htm", {"request": request})

@app.get("/top.htm", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("/top.htm", {"request": request})

@app.get("/left.htm", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("/left.htm", {"request": request})

@app.get("/blah.htm", response_class=HTMLResponse)
async def home(request: Request):
    trace1 = {
        "x": [1, 2, 3, 4],
        "y": [10, 15, 13, 17],
        "type": "scatter"
    }

    trace2 = {
        "x": [1, 2, 3, 4],
        "y": [16, 5, 11, 9],
        "type": "scatter"
    }

    data = [trace1, trace2]
    return templates.TemplateResponse("/blah.htm", {"request": request,  "data": data})

if __name__ == "__main__":
    uvicorn.run(
        "main_app:app",
        host='localhost',
        port=50805,
        reload=True
    )
