from flask import Flask, request, render_template
from src.pipelines.pred_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict_datapoint(): 
    if request.method == 'GET': 
        return render_template('form.html')
    else: 
        data = CustomData(
            month=request.form.get('month'),
            day=int(request.form.get('day')),
            order=int(request.form.get('order')), 
            country=request.form.get('country'), 
            sessionID=int(request.form.get('sessionID')),
            page1_main_category=request.form.get('page1_main_category'), 
            page2_clothing_model=request.form.get('page2_clothing_model'), 
            colour=request.form.get('colour'), 
            price=float(request.form.get('price'))
        )

        new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(new_data)

        results = round(pred[0], 2)

        return render_template('results.html', final_result=results)

if __name__ == '__main__': 
    app.run(host='0.0.0.0', debug=True)
#http://127.0.0.1:5000/ in browser