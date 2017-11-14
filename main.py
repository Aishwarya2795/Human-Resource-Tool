from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def index():
    return render_template("index.html")

@app.route('/tables.html')
def tables():
    return render_template("tables.html")

@app.route('/model1.html')
def model1():
    return render_template("model1.html")

@app.route('/model2.html')
def model2():
    return render_template("model2.html")

@app.route('/model3.html')
def model3():
    return render_template("model3.html")

@app.route('/model4.html')
def model4():
    return render_template("model4.html")

@app.route('/predict.html')
def predict():
    return render_template("predict.html")

@app.route('/predict.html', methods=['POST'])
def predicts():
        satisfaction_level = request.form['satisfaction_level']
        last_evaluation = request.form['last_evaluation']
        number_of_projects = request.form['number_of_projects']
        average_monthly_hours = request.form['average_monthly_hours']
        years_at_company = request.form['years_at_company']
        had_work_accident = request.form['had_work_accident']
        promotion_in_last_5_years = request.form['promotion_in_last_5_years']
        department = request.form['department']
        salary = request.form['salary']
        form = True
        return render_template("predict.html", satisfaction_level=satisfaction_level, last_evaluation=last_evaluation, number_of_projects=number_of_projects, average_monthly_hours=average_monthly_hours, years_at_company=years_at_company, had_work_accident=had_work_accident, promotion_in_last_5_years=promotion_in_last_5_years, department=department, salary=salary, form=form)

if __name__ == "__main__":
    app.debug = True
    app.run()