# Bank-Churn-Prediction-using-Deep-Learning-And-Machine-Learning-Models

## Supervisor for this project carried out by talented students of BITS Karyn Anselm D'souza, Vinayak Sengupta, Shaik Sabiha

Churn Rates (customers leaving or closing accounts) in companies for various reasons have also as result become a rising concern. Hence, the retention of customers has become paramount. According to the Qualtrics Banking Report, “poor service” was the reason the customers left. 56% of these customers stated that with better services they could have been retained. According to some customers they felt unvalued by the banks themselves. In a customer centric industry, customer loyalty is the decider of all, and customer experience invites loyalty. Research at Bain & Company suggested that a 5% decrease in customer population can reduce profits from 95% to 25%. 

The dataset containing 10000 rows and 14 columns was considered. This dataset was obtained from Kaggle.  This dataset contains information about the card holder in a bank and their possibility of churn. The attributes in the dataset are customer_id, credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary, and churn.  A value of 1 for churn indicates that the member has exited while 0 indicates the member is still using the bank. 

https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling

Below is a tabular representation of the dataset contents:

![alt text](https://github.com/siddhaling/Bank-Churn-Prediction-using-Deep-Learning-And-Machine-Learning-Models/blob/main/1.jpg)

The contents of each column are as as below:

![alt text](https://github.com/siddhaling/Bank-Churn-Prediction-using-Deep-Learning-And-Machine-Learning-Models/blob/main/2.jpg)

## Methodology
As we have established before that customer churn is a serious threat to the lucrativity and success of financial businesses in the consumer-driven industry. Today leading consultancies and fin-tech companies are investing heavy research and resources into tackling this issue in terms of both prevention and the cure. Our research dwelves into tackling churn in the prevention process, where we are predicting the probability of a current customer churn based on their overall financial records. The methodology is two-folded firstly we create a prediction model and test it and lastly, conduct an analysis of the various attributes of customers that contribute to churn today. 
![alt text](https://github.com/siddhaling/Bank-Churn-Prediction-using-Deep-Learning-And-Machine-Learning-Models/blob/main/3.jpg)

## Results
The figures below depict the count plots for various numeric attributes.
![alt text](https://github.com/siddhaling/Bank-Churn-Prediction-using-Deep-Learning-And-Machine-Learning-Models/blob/main/4.jpg)

We can infer the following things from the above images:

    • Out of the three countries whose data was collected, Germany had the maximum churn rate compared to France and Spain.
    • When it comes to Gender, we can see that overall Females have a greater churn capacity than Males.
    • We can also see that the people who have a credit card have a greater churn rate than the people who don’t have a credit card.
    • Finally we see that those members who are active stay with the bank and don’t churn as much as compared to those members who are inactive.
 
The total number of people who have exited and those who have been retained by the bank.

![alt text](https://github.com/siddhaling/Bank-Churn-Prediction-using-Deep-Learning-And-Machine-Learning-Models/blob/main/5.jpg)

As we can see from the above pie chart,the bank was able to retain 79.6% of the customers while 20.4% did exit the bank.

Next, we have also plotted a KDE plot which is used for visualizing the Probability Density of a continuous variable, in our case the Balance amount of the customers in the bank.

![alt text](https://github.com/siddhaling/Bank-Churn-Prediction-using-Deep-Learning-And-Machine-Learning-Models/blob/main/6.jpg)

The relationship better among all the variables a heatmap was plotted as shown below:

![alt text](https://github.com/siddhaling/Bank-Churn-Prediction-using-Deep-Learning-And-Machine-Learning-Models/blob/main/7.jpg)

In a heat map, each square shows the correlation between the variables on each axis. Values closer to zero means there is no linear correlation between the two variables for example between balance and credit_score , estimated_salary and tenure etc. The values that are close to 1 are more positively correlated showcasing good dependency on each other for example the age and churn and balance and churn etc.  A correlation closer to -1 indicates that the two variables are inversely proportional like the balance and product_number , active_member and churn etc.For the rest the larger the number and darker the colour the higher the correlation between the two variables. 

The following table shows the top three variables contributing to churn namely Age, Balance and Estimated_Salary whose inference was drawn from the heatmap.

![alt text](https://github.com/siddhaling/Bank-Churn-Prediction-using-Deep-Learning-And-Machine-Learning-Models/blob/main/8.jpg)

The graphical representation of the above table with the following box plots.

![alt text](https://github.com/siddhaling/Bank-Churn-Prediction-using-Deep-Learning-And-Machine-Learning-Models/blob/main/9.jpg)

We can further study the relationship between the top two variables contributing to churn namely Age and Balance with the help of a scatter plot as shown below.

![alt text](https://github.com/siddhaling/Bank-Churn-Prediction-using-Deep-Learning-And-Machine-Learning-Models/blob/main/10.jpg)

We can infer from the above scatter plot that the range of the majority churn age of various customers of the bank lie between 42 and 60 and the range of most of the customer’s balances were between 5000 and 150000.

We have trained ANN over the dattbase for 100 epochs. We are able to get the accuracy as well as the loss with each dataset which is shown as a graphical representation below.

![alt text](https://github.com/siddhaling/Bank-Churn-Prediction-using-Deep-Learning-And-Machine-Learning-Models/blob/main/11.jpg)

As we can infer from the above line chart, our model hits an overall of 84.5 % accuracy on the ability to predict the probability that a client will churn from the bank or not from the dataset.

The loss of our model is around 0.36 as shown in the below line chart. Unlike accuracy, loss is not a percentage. It is a summation of the errors made for each example in the training or validation set. 

![alt text](https://github.com/siddhaling/Bank-Churn-Prediction-using-Deep-Learning-And-Machine-Learning-Models/blob/main/12.jpg)

Following Table also highlights various other inferences:

![alt text](https://github.com/siddhaling/Bank-Churn-Prediction-using-Deep-Learning-And-Machine-Learning-Models/blob/main/13.jpg)

The ROC curve is plotted with the True Positive Rate on the y-axis and the False Positive Rate along the x-axis. The AUC(Area under Curve) values lies between 0.5 to 1, where 0.5 denotes an underperforming classifier and 1 denotes a very strong classifier. As we can see from the below ROC curve, our model AUC has a value of 0.80 which means that there is 80% chance that our model will be able to distinguish between the churn and the no churn classes.

![alt text](https://github.com/siddhaling/Bank-Churn-Prediction-using-Deep-Learning-And-Machine-Learning-Models/blob/main/14.jpg)

# Further Projects and Contact
www.researchreader.com

https://medium.com/@dr.siddhaling

Dr. Siddhaling Urolagin,\
PhD, Post-Doc, Machine Learning and Data Science Expert,\
Passionate Researcher, Focus on Deep Learning and its applications,\
dr.siddhaling@gmail.com

