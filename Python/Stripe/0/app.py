from flask import *
import stripe
from stripe.api_resources import payment_method

app = Flask(__name__)
app.config[
    "STRIPE_PUBLIC_KEY"
] = "pk_test_51Hdf6IJzMECqGOD86djVmO4RmD2d1kKPHzxFWSN3koXkPcUDeusLHdx7ls7ZMmjyg12edFvDD9ODMKlJmlWfWpGa00AsOAIFuT"
app.config[
    "STRIPE_SECRET_KEY"
] = "sk_test_51Hdf6IJzMECqGOD8M4GG7fkfxyKVCT52KSrSmMbas5iRW8baYKh3CmlQGFlV5wE3tx2Z8CvF5GLiHv8YyXAyW2w800d1zcUezW"
stripe.api_key = app.config["STRIPE_SECRET_KEY"]


@app.route("/")
def index():
    session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[{
            'price': 'price_1JIlMAJzMECqGOD8Pj0orIPH',
            'quantity': 1,
        }],
        mode='payment',
        success_url=url_for('thanks', _external=True) + '?session_id={CHECKOUT_SESSION_ID}',
        cancel_url=url_for('index', _external=True),
    )
    return render_template(
        "index.html",
        checkout_session_id=session["id"],
        checkout_public_key=app.config["STRIPE_PUBLIC_KEY"],
    )


@app.route("/thanks")
def thanks():
    return render_template("thanks.html")


if __name__ == "__main__":
    app.run(debug=True)
