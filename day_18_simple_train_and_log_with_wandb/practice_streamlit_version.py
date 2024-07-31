import altair as alt
import pandas as pd
import streamlit as st
import time

# forward
def predict(x, w, b):
    return x * w + b

def gradient(y_hat, y, x):
    dw = 2 * x * (y_hat - y)
    db = 2 * (y_hat - y)

    return (dw, db)

def compute_loss(y_hat, y):
    return (y_hat - y)**2 / 2.0

# update weights
def update_weights(w, b, lr, dw, db):
    w_new = w - lr * dw
    b_new = b - lr * db

    return (w_new, b_new)

def main():
    areas = [6.7, 4.6, 3.5, 5.5]
    prices = [9.1, 5.9, 4.6, 6.7]

    dataset = pd.DataFrame({
        'areas': areas,
        'prices': prices
    })

    b = 0.04
    w = -0.34
    lr = 0.01
    epochs = 10

    X_train = dataset['areas']
    Y_train = dataset['prices']

    N = len(X_train)

    loss_data = pd.DataFrame({
        "steps": [float(0)],
        "losses": [float(100)]
    })

    st.title("Model Train Monitoring")

    st.write("## Data chart:")
    st.line_chart(dataset, x="areas", y="prices")
    st.write("## Loss tracking:")

    step = 1

    loss_chart = st.line_chart(loss_data, x="steps", x_label="Steps", y="losses", y_label="Loss")

    for _epoch in range(epochs):
        for i in range(N):
            x = X_train[i]
            y = Y_train[i]

            y_hat = predict(x, w, b)
            loss = compute_loss(y_hat, y)

            new_pf = pd.DataFrame({
                "steps": [float(step)],
                "losses": [float(loss)]
            })

            loss_chart.add_rows(new_pf)

            (dw, db) = gradient(y_hat, y, x)
            (w, b) = update_weights(w, b, lr, dw, db)

            step += 1

if __name__ == "__main__":
    main()
