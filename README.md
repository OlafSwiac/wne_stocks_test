# Trading Algorithm based on ML models
## Author
Olaf Swiac
## Jak pracowac z kodem
Aby odpalić program:  
* podać odpowiedni słownik ze strategią do włącznienia, która zawiera najważniejsze parametry
* włączyć trading_main.py
* utworzy to plik csv z wynikami w folderze results_csv o podanej nazwie

Generacja comiesięcznych zmian spółek:  
* zmienić zmienną generate_list_of_stocks = True
* wybrać odpowiednią metrykę (odkomentować jedną linijkę z investment_back[stock] = ...) oraz zakres wybieranych spółek w pliku trading_enviroment.py w funckji update_stocks
* podać nazwę pliku w trading_initialize.py
* wyniki zapiszą się pod daną nazwą jako słownik z kluczami jako numerami poszczególnych okresów oraz wartościami jako listami spółek

Plik metrics.py zawiera funkcje z zaimplementowanymi metrykami (możliwe zmiany przy ARC w ilości miesięcy - nie jest to to samo co liczba okresów).

Plik DowloadStockData.py w folderze Additional_files służy do pobrania wszystkich danych z yfinance.

Plik helper.py w folderze Additional_files zawiera kod, który tworzy słownik ze wszystkimi wynikami -> używany do tworzenia tabel oraz wykresów.
