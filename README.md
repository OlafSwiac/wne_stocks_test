# Trading Algorithm based on ML models
## Author
Olaf Swiac
## Short project description
Currently trained on 4y period with the values for the day predicted based on previous 20  
Used stocks: 
* MSFT
* NKE
* INTC
* AAPL
* GOOGL
* AMZN
* GME
* AMD



Initial results (to correct) with starting balance $100 000,
models:
* SVR
* Lasso
* BayesianRidge

Combined into one prediction, each model with weight 1/3

![Ensamble - day_of_losing](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/20c98176-2e4c-416f-9dd2-076bf92aab6c)

![Ensamble - day_of_losing - number of stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/94de1d4b-917e-445c-8759-8f23ba41555e)

![Ensamble - day_of_losing - $ in stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/4cb7f340-b1a7-46d2-a66b-a8ca7553e45b)

## Current problems:
* The amount of each stock stabilizes after some time - posibility in adding randomness --> ADDED, to check
* Adding an algorithm for picking the best variables in the model
