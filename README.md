# Lenovo Predicting Customer Sentiment

Through this project an attempt has been made to help Lenovo predict the customer satisfaction for their products with the help of customer sentiment data. Net Promoter Score (NPS) is a key indicator of a product or company’s performance in the market.

## **Problem Statement**: 
Lenovo currently uses customers’ responses to the Net Promoter Score (NPS) survey as their primary measurement of customer satisfaction, but feedback from the NPS survey results arrives rather late in the product life cycle. 

## **Objective**: 
To address the customers’ concerns in a timely fashion, Lenovo would like to understand them based on the discussion customers have on the web regarding their products. They would like to be able to predict NPS score based on the customer sentiment.

Having an NPS score would give an idea of where the product, and the company in general, stands among the public. Higher Net Promoter Scores tend to indicate a healthier business, while lower Net Promoter Scores can be an early warning to dig deeper into potential customer satisfaction and loyalty issues.

## **Data:**
The following key information was provided by Lenovo to help to identify the relationship between customer sentiment and NPS:
* Sentiment Data: Information collected from web scrapes and third-party apps which had data regarding the following:
	- Comment ID
	- Sentiments that the software recognized and allotted to various product features in that comment
	- Levels of classification of the product features
	- Product name
	- Date (MM/DD/YYYY)
	- Star Rating assigned by the customer
* Survey Data: Official survey responses to many questions important among which are:
	- Survey ID & date (MM/DD/YYYY)
	- Product & Brand NPS
	- Product name & Series
	- Additional questions like satisfaction levels with various features, service aspects, purchase process, etc.

## **Methodology:**

### Markov Decision Process:

A Markov decision model was developed to look at the evolution of sentiment over time and to suggest any intervention that would be beneficial.  Given the inherent uncertainties in the existing process, a stochastic model would accurately capture the development of sentiment.


### Variable selection:

*	Taxonomies as predictors: 

	To understand how customer sentiments, affect the pNPS, it was decided that the frequency of sentiments be used as predictors of pNPS. In order to capture the variance between taxonomies, the frequency of sentiments was broken down at a taxonomy level to be used for further analysis. Because of the high multicollinearity between taxonomies, grouping of taxonomies based on correlation and class of taxonomy was done.

<img src="https://github.com/SatyajitNarayanan/Lenovo_Predicting_Customer_Sentiment/raw/master/images/Variable_selection.JPG" alt="Poster Image">


* Evolution of Sentiment as a predictor: (Why, how, Examples)

	To understand the impact of development of sentiment over time on the pNPS, an Evolution of Sentiment (EOS) variable was created. The transition probability matrix for each product series was computed in a manner similar to P-matrix of the MDP model. Then the sentiment distribution for each series in the first month (q) was computed from the data available and the distribution in the following months was obtained using the Chapman-Kolmolgorov rule (q<sup>n</sup> = q*P<sup>n</sup> where q<sup>n</sup> = sentiment distribution in nth epoch).

	To calculate EOS, the distribution of sentiment in the first month should be known. The sentiment distributions across various months are then used as independent variables in the regression analysis.

<img src="https://github.com/SatyajitNarayanan/Lenovo_Predicting_Customer_Sentiment/raw/master/images/EOS_Calculation.JPG" alt="Poster Image">



### **Prediction:**

**Principal Component Analysis (PCA):**

In order to eliminate issues like multicollinearity and sub optimal solutions, principle component analysis (PCA) was used to figure out values of set of linearly independent uncorrelated variables called principle components. PCA eliminated these two issues due to the orthogonal transformation property of the principle components. The components were then used as regressors to perform principle component regression. In the present context, each product type (Consumer and Commercial) had multiple taxonomy levels.

<img src="https://github.com/SatyajitNarayanan/Lenovo_Predicting_Customer_Sentiment/raw/master/images/PCA.JPG" alt="Poster Image">

**Calculation of pNPS as a response variable:**

The pNPS value for a month is calculated as cumulative subtraction of promoter % and detractor % till that month. It is assumed that the sentiments will reflect on the pNPS after a lag of five months. Also, Five-month lag provides maximum overlap between sentiment and survey data.

After deciding the dependent and independent variables, the level of data to be used for prediction needed to be decided. Data at a Series-Month level was considered. Independent and dependent variables were calculated at this level each for Consumer and Commercial data.


### **Results:**

<img src="https://github.com/SatyajitNarayanan/Lenovo_Predicting_Customer_Sentiment/raw/master/images/Results.JPG" alt="Poster Image">

