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

### Added stop los
2010 -> start of training, "trading" for 10 * 2 months
Sharpe and Sortino ratio should be lower -> currently multiplying by sqrt(252) but I am only trading on about a half of it -> waiting every 2 month 20 trading days to collect data

![Dziwny wynik - najlepszy](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/7d605446-6d70-41c0-a241-134296757c70)

![Dziwny wynik - najlepszy - stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/b0972eee-8e60-45e1-bfc2-724325668713)

![Dziwny wynik - najlepszy - ratios](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/e0b0b240-1169-48a7-8e8e-4f5081979e6f)

![Dziwny wynik - najlepszy - $ in stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/e815af1c-02b7-4050-b268-5925cf695a4d)

2011 -> start of training, "trading" for 14 * 2 months

![2011 - start - 14 months - stop los](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/4505e8cd-b35f-4af1-a993-9f416fd31139)

![2011 - start - 14 months - stop los - stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/0147a7c5-17ea-42c7-b83b-c472a556b58d)

![2011 - start - 14 months - stop los - $ in stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/e9da4d19-a1f1-4906-b116-19a9497cbe21)

![2011 - start - 14 months - stop los - ratios](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/b1043499-57da-4c73-83c0-1cacc634b940)

2013 -> start of training, "trading" for 14 * 2 months

![2013 - start - 14 months - stop los - $ in stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/5d65d782-c85e-4efe-ae91-539c6a07a6a0)

2014 -> start of training, "trading" for 16 * 2 months

![2014 - start - 32 months - stop los - 0 005 transaction costs](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/5404406b-07f6-4f89-a3ff-c7a74daca764)


## Current problems:
* The amount of each stock stabilizes after some time - posibility in adding randomness --> ADDED, to check
* Adding an algorithm for picking the best variables in the model
