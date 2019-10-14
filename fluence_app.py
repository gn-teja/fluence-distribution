
#

from flask import Flask, render_template, request
import logging
from fluence_energy_dist import fluence_energy_dist_over_larger_distance as fl_dist_over_large_distances

app = Flask(__name__)

comments = []

@app.route('/helloworld')
def hello_world():
    return 'Hello from Flask!'

@app.route('/')
def form():
    return render_template('form.html')


@app.route('/results', methods=['GET', 'POST'])
def submitted_form():
    if request.method == 'POST':
        P = float(request.form['P'])
        d0 = float(request.form['d0'])
        t1 = float(request.form['t1'])
        dp1 = float(request.form['dp1'])
        dh1 = float(request.form['dh1'])
        t2 = float(request.form['t2'])
        dp2 = float(request.form['dp2'])
        dh2= float(request.form['dh2'])

        FluenceInput = [P, d0, t1, dp1, dh1, t2, dp2, dh2]

        FluenceOutput, plot1, plot2, plot3, plot4, plot5 = fl_dist_over_large_distances(P,d0,t1,dp1,dh1,t2,dp2,dh2)

        # FluenceOutput = [338.8806288286047, 325.77040999339755, 274.6904164261084, 274.6904164261084, 98.87079383514575, 74.1530953763593, 399.24154764375083, 160.40100250626568, 238.8059701492537]

        # P,d0,t1,dp1,dh1,t2,dp2,dh2
        # 200,406,80,95,105,60,67,75
    else:
        FluenceInput = None
        FluenceOutput = None
        plot1 = None
        plot2 = None
        plot3 = None
        plot4 = None
        plot5 = None

    return render_template('submitted_form.html',FluenceInput=FluenceInput,FluenceOutput=FluenceOutput, plot1=plot1, plot2=plot2, plot3=plot3, plot4=plot4, plot5=plot5)

@app.errorhandler(500)
def server_error(e):
    # Log the error and stacktrace.
    logging.exception('An error occurred during a request.')
    return 'An internal error occurred.', 500


if __name__ == 'main':
    app.run(debug=True)