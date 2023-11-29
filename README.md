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

### Added stop loss -> 2%
2010 -> start of training, "trading" for 10 * 2 months
Sharpe and Sortino ratio should be lower -> currently multiplying by sqrt(252) but I am only trading on about a half of it -> waiting every 2 month 20 trading days to collect data

![Dziwny wynik - najlepszy](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/7d605446-6d70-41c0-a241-134296757c70)

![Dziwny wynik - najlepszy - stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/b0972eee-8e60-45e1-bfc2-724325668713)

![Dziwny wynik - najlepszy - $ in stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/e815af1c-02b7-4050-b268-5925cf695a4d)

![Dziwny wynik - najlepszy - ratios](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/e0b0b240-1169-48a7-8e8e-4f5081979e6f)

Sharpe ratio: 2.4657  
Sortino ratio: 6.5806  
(corrected)

2011 -> start of training, "trading" for 14 * 2 months

![2011 - start - 14 months - stop los](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/4505e8cd-b35f-4af1-a993-9f416fd31139)

![2011 - start - 14 months - stop los - stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/0147a7c5-17ea-42c7-b83b-c472a556b58d)

![2011 - start - 14 months - stop los - $ in stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/e9da4d19-a1f1-4906-b116-19a9497cbe21)

![2011 - start - 14 months - stop los - ratios](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/b1043499-57da-4c73-83c0-1cacc634b940)

Sharpe ratio: 1.5899 
Sortino ratio: 3.9761  
(corrected)

2013 -> start of training, "trading" for 14 * 2 months

![2013 - start - 14 months - stop los - $ in stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/5d65d782-c85e-4efe-ae91-539c6a07a6a0)

2014 -> start of training, "trading" for 16 * 2 months

![2014 - start - 32 months - stop los - 0 005 transaction costs](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/5404406b-07f6-4f89-a3ff-c7a74daca764)

2014 -> start of training, "trading" for 16 * 2 months, 17 stocks

![2014 - start - 32 months - 17 stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/346cc3af-539b-441d-ba3f-2679ef5b7afa)

![2014 - start - 32 months - 17 stocks - number of stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/96187dc6-5ed3-4ccf-b531-10ea0ec7ad08)

![2014 - start - 32 months - 17 stocks - $ in stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/03577b1b-f71d-4f69-861c-c880a3381cea)

Sharpe ratio: 2.2292  
Sortino ratio: 5.9985 
(corrected)

![2010 - start - 64 months - 17 - stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/a051c35f-c8c4-42e8-83f9-8241ae9ecf12)

![2010 - start - 64 months - 17 - stocks - number of stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/0712c1fd-6485-4f17-ab4f-7af66948dc04)

![2010 - start - 64 months - 17 - stocks - $ inf stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/85531ec9-ff81-4add-8f14-bfbc986e775c)
  
Sharpe ratio: 1.256  
Sortino ratio: 3.341  

## Current problems:
* The amount of each stock stabilizes after some time - posibility in adding randomness --> ADDED, to check
* Adding an algorithm for picking the best variables in the model
  

2005 - 64 months  
![2005 - start - 64 months - 17 - stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/051f33ad-0d64-471d-b9bf-fd89c3e0bfda)

![ELON HATES ME but stocks - wtf -7000 stocks](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/108587b9-d961-4565-ab7e-1a97319c8496)

![ELON HATES ME](https://github.com/OlafSwiac/wne_stocks_test/assets/119978172/c0157e84-2790-45a6-a81b-a5692c84e1b8)


