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
* TSLA
* META
* SMCI
* BRK-B
* LLY
* TSM
* UNH
* WMT
* MA
* JNJ
* AMGN


Initial results (to correct) with starting balance $100 000,
models:
* SVR
* Lasso
* BayesianRidge

Combined into one prediction, each model with weight 1/3

![Ensamble - balance 2 (lasso)](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/ed151878-c74e-4fd5-ae3b-d9e0c748bfe0)



## Current problems:
* The amount of each stock stabilizes after some time - posibility in adding randomness --> ADDED, to check
* Adding an algorithm for picking the best variables in the model
