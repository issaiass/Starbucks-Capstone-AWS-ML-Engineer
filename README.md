# Starbucks Capstone Project


<p align="center">
  <img src="doc/imgs/starbucks.gif?raw=true" width="85%" style="display:inline-block;"/>
</p>

In this capstone of AWS Machine Learning Engineer we developend a prediction for future events (and also we could predict future acceptance proability) using machine learning techniques doing the ML Pipeline of Data Scientists..



<details open>

<summary> <b>Assignments<b></summary>


#### Objective: 
  - Our approach was to predict the next event or probabilities of the next event to ocurr.  For that, we grouped the persons and shifted the event (target) to the future and in this way try to predict next ocurrence probability.

  - Our base model was an XGBoostClassifier, at first, we test the model metrics and we decided to **push up the lowest metric**, that is, **precision of the offers completed**. The metrics of the base model pushed from a value of 27% to The final model of 28% (too low but we had to test our approach).  Our final goal was **recall** because we expected that cost more a False Negative than a false positive. Our last recall metrics were from 89% for the base model of offer completed and in the final model 71%.

  - Given this previous metrics is better to put in service the base model or get a long way to get the best hyperparameters for the model that could surpass the predictions but we test the trained model to achive the workflow for later put the base model on service.

  - Several tools were used to achieve this goal like such as Amazon SageMaker, S3, IAM and AWS Lambda, to build a end-to-end solution.

<p align="center"> </p>
</details>

<details open>
<summary> <b>Report<b></summary>

- We started our data preparations over the provided jsons of transcript, profile and event.
- After merging the data we do some inspection and analysis and concluded this:
- Late we do a analysis of the offers and we concluded this.

- First we analyzed one person and looked that:
  - A female, near mid's 40's with a very direct relationship with Starbucks coffe :)
  - By simple inspection analyzing just one person we can get that he likes the discount offer and always completes
  - She is a very loyalty person to starbucks as a "semi-senior" member with +3 years of membership
  - Mostly we need three channels to offer him (a mid 40's person)
  - Even if the reward is difficult, he tries to complete it.


- Next we proceed to analyze by groups:
  - Data Exploration by Age by Type of Offer...
     <p align="center">
      <img src="doc/imgs/eda/data_exploration_age.PNG?raw=true" width="55%" style="display:inline-block;"/>
     </p>
    - ages are around 20 to 100
    - outliers over 120
    - we could see ages between 45 to 65 are more attracted to the offers
    - specially bogo and discount are the most popular
  - Data Exploration by Income by Type of Offer...
    - we can see people that have a regular income above 25_000 and 100_000 are generally more attracted to the offers
    - a few people get the informational offer but are more uniform 
     <p align="center">
      <img src="doc/imgs/eda/data_exploration_income.PNG?raw=true" width="55%" style="display:inline-block;"/>
     </p>
  - Data Exploration by Reward in gender and by duration in gender:
    - between ages are very well distributted the offers
    - reward ranges are superiof for bogo and less rewarded for informational (they are not rewarded), that explains the popularity of bogo and discount
    - people like more the discount that have more difficulty level surpassing the 10
    - off curse because the duration is higher for the discount is very popular with bogo
    - time is well balanced between the offer types so we can clearly say we are fair between offers
     <p align="center">
      <img src="doc/imgs/eda/data_exploration_reward.PNG?raw=true" width="55%" style="display:inline-block;"/>
      <img src="doc/imgs/eda/data_exploration_duration.PNG?raw=true" width="55%" style="display:inline-block;"/>
     </p>

  - After the data exploration we proceed to train our base model on our machine and we get good results as listed below
    <p align="center">
      <img src="doc/imgs/training/base_model_test.PNG?raw=true" width="55%" style="display:inline-block;"/>
    </p>

- We made an HP Tunning Job over sagemaker, 12 models with a wide range of parameters, but for getting a better model we need to test a wide range of models with parameters, due to limit of money this is not achieved right now.
    <p align="center">
      <img src="doc/imgs/sagemaker/sagemaker_btjob.PNG?raw=true" width="100%" style="display:inline-block;"/>
      <img src="doc/imgs/sagemaker/sagemaker_hp.PNG?raw=true" width="100%" style="display:inline-block;"/>      
    </p>
- Prepared a final endpoint for invocations and test
    <p align="center">
      <img src="doc/imgs/sagemaker/sagemaker_ep.PNG?raw=true" width="100%" style="display:inline-block;"/>
    </p>

- Built a lambda functions to test our endpoint
    <p align="center">
      <img src="doc/imgs/lambda/lambda_starbucks.PNG?raw=true" width="100%" style="display:inline-block;"/>    
      <img src="doc/imgs/lambda/lambda_function.PNG?raw=true" width="100%" style="display:inline-block;"/>
      <img src="doc/imgs/lambda/lambda_event.PNG?raw=true" width="100%" style="display:inline-block;"/>
      <img src="doc/imgs/lambda/lambda_response.PNG?raw=true" width="100%" style="display:inline-block;"/>
    </p>   
- Made and API Gateway and linked to the lambda
    <p align="center">
      <img src="doc/imgs/lambda/lambda_apigw.PNG?raw=true" width="100%" style="display:inline-block;"/>    
      <img src="doc/imgs/apigw/apigw_main.PNG?raw=true" width="100%" style="display:inline-block;"/>
      <img src="doc/imgs/apigw/apigw_integration.PNG?raw=true" width="100%" style="display:inline-block;"/>
      <img src="doc/imgs/apigw/apigw_response.PNG?raw=true" width="100%" style="display:inline-block;"/>
    </p>   

- We tested our rest API with curl commands too
    <p align="center">
      <img src="doc/imgs/apigw/apigw_curl.PNG?raw=true" width="100%" style="display:inline-block;"/>    
    </p>   
    -You can copy the command from here

        curl -X POST https://0lrtdebg9i.execute-api.us-east-1.amazonaws.com/dev/api-ep \
        -H "Content-Type: application/json" \
        -d '[
        {
            "person": 2301,
            "time": 30,
            "amount": 13.25,
            "gender": "F",
            "age": 63,
            "income": 93000,
            "reward": 5,
            "difficulty": 20,
            "duration": 10,
            "offer_type": "discount",
            "email": 1,
            "social": 0,
            "mobile": 0,
            "web": 1,
            "days_since_membership": 1875,
            "offered_channels_count": 2
        },
        {
            "person": 10,
            "time": 168,
            "amount": 12.50,
            "gender": "M",
            "age": 70,
            "income": 50000,
            "reward": 3,
            "difficulty": 10,
            "duration": 5,
            "offer_type": "bogo",
            "email": 0,
            "social": 1,
            "mobile": 1,
            "web": 0,
            "days_since_membership": 1771,
            "offered_channels_count": 2
        }
        ]'

      
- Final results of our model where promissing but needs to improve due to we only trained 12 models.  For a first try, our recommendation is to put the base model because was with the best benchmarchs.
    <p align="center">
      <img src="doc/imgs/training/end_model_test.PNG?raw=true" width="100%" style="display:inline-block;"/>    
    </p>   
- A recommendation to improve is to try to use a GAN to generate more data of completed offers and generate statistically more data on the lower 5% and upper 95% (extreme outliers) to help the model to detect those frontier cases

</details>



<details open>
<summary> <b>Results <b></summary>

#### Video of the explanations must be here

</details>





<details open>
<summary> <b>Issues<b></summary>

- No issues encountered

</details>

<details open>
<summary> <b>Future Work<b></summary>

- No Future works.

</details>

<details open>
<summary> <b>Contributing<b></summary>

Your contributions are always welcome! Please feel free to fork and modify the content but remember to finally do a pull request.

</details>

<details open>
<summary> :iphone: <b>Having Problems?<b></summary>

<p align = "center">

[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/riawa)
[<img src="https://img.shields.io/badge/telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white"/>](https://t.me/issaiass)
[<img src="https://img.shields.io/badge/instagram-%23E4405F.svg?&style=for-the-badge&logo=instagram&logoColor=white">](https://www.instagram.com/daqsyspty/)
[<img src="https://img.shields.io/badge/twitter-%231DA1F2.svg?&style=for-the-badge&logo=twitter&logoColor=white" />](https://twitter.com/daqsyspty) 
[<img src ="https://img.shields.io/badge/facebook-%233b5998.svg?&style=for-the-badge&logo=facebook&logoColor=white%22">](https://www.facebook.com/daqsyspty)
[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/riawe)
[<img src="https://img.shields.io/badge/tiktok-%23000000.svg?&style=for-the-badge&logo=tiktok&logoColor=white" />](https://www.linkedin.com/in/riawe)
[<img src="https://img.shields.io/badge/whatsapp-%23075e54.svg?&style=for-the-badge&logo=whatsapp&logoColor=white" />](https://wa.me/50766168542?text=Hello%20Rangel)
[<img src="https://img.shields.io/badge/hotmail-%23ffbb00.svg?&style=for-the-badge&logo=hotmail&logoColor=white" />](mailto:issaiass@hotmail.com)
[<img src="https://img.shields.io/badge/gmail-%23D14836.svg?&style=for-the-badge&logo=gmail&logoColor=white" />](mailto:riawalles@gmail.com)

</p

</details>

<details open>
<summary> <b>License<b></summary>
<p align = "center">
<img src= "https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-sa.svg" />
</p>
</details>