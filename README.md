# forex_bot
machine learning forex predictive tool

Hi All welcome to my forex predictive tool
This model predicts FX price action 2 time frames away using the Support Vector Machine model in sci-kit learn
The process of the scrript is explained in the script itself
 - in summary it downloads historical fx data and texts the correlation of technical price indicators
 - once the best set technical indicators are found it then attempts to find the optimal combination of these indicators
 - once the best combination has been found it then looks to tune the machine for the optimal parameter settings
 - at the end the optimised results are printed to an excel spreadsheet
 
Reqs outside of the requirements.txt file, excel, SQL (and an application that can open an SQL data base)
A metatrader account (i used a free demo account) with account user and password in creds/creds.txt
I created my free demo account using this website: https://admiralmarkets.com/login?
