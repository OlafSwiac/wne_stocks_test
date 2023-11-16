# Trading Algorithm based on SVR model
## Author
Olaf Swiac
## Short project description
Currently trained on 4y period with the values for the day predicted based on previous 15  
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


Initial results (to correct) with starting balance $100 000:  

  ![WNE_STOCKS_RESULTS_1](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/3e405dcb-2747-42ac-a67d-d4efb6730478)
  ![WNE_STOCKS_RESULTS_2](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/44c6374b-25ac-44a8-9f3d-5ab07e413168)


## Current problems:
* The amount of each stock stabilizes after some time - posibility in adding randomness
* One day of 0 portfolio value
* Adding an algorithm for picking the best variables in the model
* Finding other "good" models for Ensemble Model   
