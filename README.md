---
title: "Head Start for Data Scientist"
output:
  html_document:
    number_sections: TRUE
    toc: TRUE
    fig_height: 4
    fig_width: 7
    code_folding: show
---
[原文地址](https://www.kaggle.com/hiteshp/head-start-for-data-scientist)

```r
#--显示R代码  knitr包的opts_chunk$set()函数可以配置 隐藏或者显示代码 
knitr::opts_chunk$set(echo=TRUE)
```

早期，当我是一个机器学习的新手  
我曾经不知所措，比如选择语言进行编码，选择正确的在线课程，或选择正确的算法。  
所以，我打算让人们更容易进入机器学习。  
我会假设我们中的很多人都是从机器学习的旅程开始的。 让我们来看看当前的专业人员如何达到目标，以及如何模仿他们。  


#### 第一阶段  保证独立完成  
对于刚开始使用机器学习的人来说，把自己和学习，教授和练习机器学习的人们联系在一起非常重要。  
独自学习是不容易的。 所以，保证要自己去学习机器学习，可以找到数据科学论坛社区来帮助你减少困难。  


#### 第二阶段  学习生态系统(意思就是进圈，和娱乐圈差不都，就是一个人文环境)  
发现机器学习的生态系统  
数据科学是一个充分利用开源平台的领域。 虽然数据分析可以用多种语言进行，但使用正确的工具可以制定或中断项目。    
数据科学图书馆在Python和R生态系统中蓬勃发展。 在这个网址查看 用Python还是R进行数据分析。[链接](https://www.datacamp.com/community/tutorials/r-or-python-for-data-analysis)  
无论您选择哪种语言，Jupyter Notebook和RStudio都让我们的生活变得更加轻松。 它们允许我们在操纵数据的同时可视化数据。 按照这个[链接](http://blog.kaggle.com/2015/12/07/three-things-i-love-about-jupyter-notebooks/)阅读更多关于Jupyter Notebook的功能。  
Kaggle，Analytics Vidhya，MachineLearningMastery和KDNuggets是一些活跃的社区，世界各地的数据科学家在这里丰富彼此的学习。  
机器学习已经通过在线课程或者Coursera，EdX等的MOOC进行民主化，在这里我们向世界一流大学的杰出教授学习。 这里有一个[顶级MOOC列表](https://medium.freecodecamp.org/i-ranked-all-the-best-data-science-intro-courses-based-on-thousands-of-data-points-db5dc7e3eb8e)有关于现在可用的数据科学列表。  


#### 第三阶段  巩固基础
学习操纵数据  
根据访谈和专家估计，数据科学家将50％到80％的时间用于收集和准备不规则数字数据的世俗工作，然后才能探索有用的金块。 - 纽约时报的史蒂夫·洛尔  
数据科学不仅仅是建立机器学习模型。 这也是解释模型并用它们来推动数据驱动的决策。 在从分析到数据驱动的结果的过程中，数据可视化以强有力和可信的方式呈现数据，扮演着非常重要的角色。  
Python中的[Matplotlib](https://matplotlib.org/)库或R中的[ggplot](http://ggplot2.org/)提供了完整的2D图形支持，具有非常高的灵活性，可以创建高质量的数据可视化。  
有一些图书馆您将花费大部分时间在上面当您在进行分析时。  


#### 第四阶段  日复一日的练习
学习机器学习算法并且每天练习  
有了基础之后，你可以实现机器学习算法来预测和做一些很酷的东西  
Python中的Scikit-learn库或R中的caret, e1071库通过一致的接口提供一系列有监督和无监督的学习算法。  
这些让你实现一个算法，而不用担心内部工作或细节的细节。  
将这些机器学习算法应用到您身边的用例中。这可能是在你的工作，或者你可以在Kaggle比赛中练习。 其中，全世界的数据科学家都在竞相建立模型来解决问题。  
同时了解一种算法的内部运作情况。 从机器学习的“Hello World！”开始，线性回归然后转到Logistic回归，决策树到支持向量机。 这将需要你刷新你的统计和线性代数。  
Coursera创始人Andrew Ng是AI的先驱，开发了一个[机器学习课程](https://www.coursera.org/learn/machine-learning)，为您理解机器学习算法的内部工作提供了一个很好的起点。  


#### 第五阶段  学习高级技能
了解复杂的机器学习算法和深度学习架构  
虽然机器学习作为一个领域早已建立起来，但最近的炒作和媒体关注主要是由于机器学习在计算机视觉，语音识别，语言处理等AI领域的应用。 其中许多是由Google，Facebook，微软等科技巨头开创的。  
这些最新的进展可以归功于廉价计算的进步，大规模数据的可用性以及深度学习架构的发展。  
要深入学习，您需要学习如何处理非结构化数据 - 无论是文本，图像，  
您将学习使用像TensorFlow或Torch这样的平台，这使我们可以应用深度学习，而不用担心低级别的硬件要求。 你将学习强化学习，这使得像AlphaGo Zero这样的现代AI奇迹成为可能  


 
我在Kaggle看到许多新的学习者，想为他们创造一个kernal，让他们有一个良好的开端。  
这个针对基础学习者的kernal，是试图快速理解数据科学，我选择了定期的对话方式。  
在kernal中，我们将遇到两个字符“MARK”和“JAMES”，其中MARK是Data Science（Laymen）的新成员，JAMES使他理解概念  
为了容易的开始 我选择了泰坦尼克号的数据集  


#### 数据集简介
On 14 April 1912, the [RMS Titanic](https://en.wikipedia.org/wiki/RMS_Titanic) struck a large iceberg and took approximately 1,500 of its passengers and crew below the icy depths of the Atlantic Ocean. Considered one of the worst peacetime disasters at sea, this tragic event led to the creation of numerous [safety regulations and policies](http://www.gc.noaa.gov/gcil_titanic-history.html) to prevent such a catastrophe from happening again. Some critics, however, argue that circumstances other than luck resulted in a disproportionate number of deaths. The purpose of this analysis is to explore factors that influenced a person’s likelihood to survive.  
上面的英文大概就是讲了一下泰坦尼克号的故事吧~  


#### 选择软件 
The following analysis was conducted in the [R software environment for statistical computing](https://www.r-project.org/).

#### 下面的讲解主要以对话的形式进行
问：今天学啥？  
答：数据科学基础  
 
问：啥是数据科学？  
答：数据科学是数据推理，算法开发和技术的多学科融合，用来解决分析复杂的问题。  
                      
问： 数据科学家如何进行数据挖掘  
答： 用这些开始  
	 1. 收集--解决问题所需的原始数据。  
	 2. 加工--数据缠绕  
	 3. 探索--数据可视化  
	 4. 展示--深入分析(机器学习，统计模型，算法)  
	 5.  交流--分析的结果  

问：能不能详写解释一下  
答：不能  


#### 导入数据集                                     
问：怎么把数据集插入到Rstudio  
答：自己百度  

问：万一调用库失败咋办  
答：自己查  

问：我要开始了好激动  
答：赶紧的吧 真墨迹  

```r
# data wrangling  数据处理包
library(tidyverse)
library(forcats)
library(stringr)
library(caTools)

# data assessment/visualizations  数据可视化包
library(DT)
library(data.table)
library(pander)
library(ggplot2)
library(scales)
library(grid)
library(gridExtra)
library(corrplot)
library(VIM) 
library(knitr)
library(vcd)
library(caret)

# model  模型包
library(xgboost)
library(MLmetrics)
library('randomForest') 
library('rpart')
library('rpart.plot')
library('car')
library('e1071')
library(vcd)
library(ROCR)
library(pROC)
library(VIM)
library(glmnet) 
```

问：现在可以导入数据集了吧  
答：嗯  
```r
train <- read_csv('../input/train.csv')
test  <- read_csv('../input/test.csv')
```

为了研究完整的数据集，可以加入测试和训练数据集。  
在此之前，我们将添加一个新的列“set”，并为测试数据集命名为“test”  
和“训练”列车数据集，以了解它是哪条记录。  
```r
train$set <- "train"
test$set  <- "test"
test$Survived <- NA
full <- rbind(train, test)
```

问：我们已经处理了用于解决问题的原始数据    
答：嗯 接着处理   
                                     
问：为什么我们需要处理数据（数据缠绕）  
答：你收集的数据目前仍是原始数据，这很可能包含错误，缺失和腐败的价值。  
在您从数据中得出任何结论之前，您需要对其进行一些数据处理，  
这是我们下一节的主题。我们选择我们想要操作的数据  
                                                                        
问：在数据科学下进行了哪些操作？  
答：这是所有的观点  
<center><img src="https://doubleclix.files.wordpress.com/2012/12/data-science-02.jpg"></center>
这个也给了一些清晰的观点  
<center><img src="https://cdn-images-1.medium.com/max/1600/1*2T5rbjOBGVFdSvtlhCqlNg.png"></center>
                                                                          
问：这看起来不错，把工具和语言都显示出来了  
答：在这里我们将使用R语言做处理                                                                           

                                                                                                                         
### 在此之前，我们需要一个数据检查 - 探索性分析（EDA） 
1. 检查数据集的疏散  
2. 数据集的维度  
3. 列名  
4. 每行有多少不同的值  
5. 缺失值  

```r
# check data 检查数据
str(full)

# dataset dimensions  维度信息
dim(full)

# Unique values per column  每列有多少种值
lapply(full, function(x) length(unique(x))) 

#Check for Missing values 检查缺失值

library(tidyverse)
library(forcats)
library(stringr)
library(caTools)
missing_values <- full %>% summarize_all(funs(sum(is.na(.))/n()))    #缺失值比例  funs 函数列表  summarize_all将函数应用于每一列
missing_values <- gather(missing_values, key="feature", value="missing_pct") #gather 转化成key-value形式 
missing_values %>%
  ggplot(aes(x=reorder(feature,-missing_pct),y=missing_pct)) +   #reorder 重新排序  默认的将第一个参数作为分类变量处理，将第二个变量重新排序 就是先分类 再排序
  geom_bar(stat="identity",fill="red")+     #geom_bar 画条形图
  coord_flip()+theme_bw()    #coord_flip()  旋转横纵坐标  横坐标变为纵坐标 纵变横   theme_bw() 添加一个坐标的主题样式
```
<img src="/img/1.png"></img>
```r
#对缺失值有用的数据质量函数
#检查 一个数据集df的某一列colname的类型  返回列的属性列表(列名，类型，总个数，缺失值个数，numInfinite，平均值，最小值，最大值)
checkColumn = function(df,colname){

  testData = df[[colname]]
  numMissing = max(sum(is.na(testData)|is.nan(testData)|testData==''),0)  #最大缺失值

  #class(x)得到x的类型
  if (class(testData) == 'numeric' | class(testData) == 'Date' | class(testData) == 'difftime' | class(testData) == 'integer'){
    list('col' = colname,'class' = class(testData), 'num' = length(testData) - numMissing, 'numMissing' = numMissing, 'numInfinite' = sum(is.infinite(testData)), 'avgVal' = mean(testData,na.rm=TRUE), 'minVal' = round(min(testData,na.rm = TRUE)), 'maxVal' = round(max(testData,na.rm = TRUE)))
  } else{
    list('col' = colname,'class' = class(testData), 'num' = length(testData) - numMissing, 'numMissing' = numMissing, 'numInfinite' = NA,  'avgVal' = NA, 'minVal' = NA, 'maxVal' = NA)
  }
}

#检查数据集的所有列属性  得到所有列属性列表的数据框
checkAllCols = function(df){
  resDF = data.frame()
  for (colName in names(df)){
    resDF = rbind(resDF,as.data.frame(checkColumn(df=df,colname=colName)))
  }
  resDF
}

#属性列表可视化
library(DT)
datatable(checkAllCols(full), style="bootstrap", class="table-condensed", options = list(dom = 'tp',scrollX = TRUE))
```
<img src="/img/1-1.png"></img>
```r
#map_dbl返回一个与输入长度相同的向量。
#round(x, n) x的约数 精确到n位
miss_pct <- map_dbl(full, function(x) { round((sum(is.na(x)) / length(x)) * 100, 1) })  #缺失值比率
miss_pct <- miss_pct[miss_pct > 0]

data.frame(miss=miss_pct, var=names(miss_pct), row.names=NULL) %>%
    ggplot(aes(x=reorder(var, -miss), y=miss)) + 
    geom_bar(stat='identity', fill='red') +
    labs(x='', y='% missing', title='Percent missing data by feature') +
    theme(axis.text.x=element_text(angle=90, hjust=1))
```
<img src="/img/2.png"></img>

#### Feature engineering.
问：什么是feature engineering  
答：该过程试图从数据中现有的原始特征创建额外的相关特征，并提高学习算法的预测能力。详情查看上面这个[网址](https://github.com/bobbbbbi/Machine-learning-Feature-engineering-techniques)


## 数据操作  数据预处理                                                                           
问：我们已经了解了我们的数据集吧  
答：然后我们还需要做一些数据的处理  
                                    
问：怎么进行数据处理  
答：数据操作是一个改变数据的过程，为了使其更容易阅读并且更有组织性  
                                                                                  
下面的部分着重于准备数据，以便用于学习训练，比如探索性数据分析和建模拟合。  

#### 处理Age字段
#-- 对于年龄的处理   将缺失值替换为平均值
```r
#-- mutate() 添加新的变量并保留现有的
full <- full %>%
    mutate(
      Age = ifelse(is.na(Age), mean(full$Age, na.rm=TRUE), Age),       #如果为空 就赋平均值
      `Age Group` = case_when(   Age < 13             ~ "Age.0012",    #case when 分组
                                 Age >= 13 & Age < 18 ~ "Age.1317",
                                 Age >= 18 & Age < 60 ~ "Age.1859",
                                 Age >= 60            ~ "Age.60Ov"))

```

#### 处理Embarked字段
#--使用常见的符号（感觉是出现次数最多的 即S）来替换 Embarked 的空值
```r
full$Embarked <- replace(full$Embarked, which(is.na(full$Embarked)), 'S')
```

#### 处理 Titles字段
#--从Name特征中提取个人标题
```r
names <- full$Name
title <-  gsub("^.*, (.*?)\\..*$", "\\1", names)
full$title <- title
table(title)
#--names---------------------------------------------------
#--[1297] Nourney, Mr. Alfred (Baron von Drachstedt")"     
#--[1298] Ware, Mr. William Jeffery                        
#--[1299] Widener, Mr. George Dunton                      
#--[1300] Riordan, Miss. Johanna Hannah""
#--[1301] Peacock, Miss. Treasteall                        
#--[1302] Naughton, Miss. Hannah                           
#--[1303] Minahan, Mrs. William Edward (Lillian E Thorpe)  
#--[1304] Henriksson, Miss. Jenny Lovisa                   
#--[1305] Spector, Mr. Woolf                               
#--[1306] Oliva y Ocana, Dona. Fermina        
#--title 分割后结果----------------------------------------
#--[1297] "Mr"           "Mr"           "Mr"          
#--[1300] "Miss"         "Miss"         "Miss"        
#--[1303] "Mrs"          "Miss"         "Mr"          
#--[1306] "Dona"         "Mr"           "Mr" 


#--MISS, Mrs, Master and Mr 是出现最多的符号  
#--更好的做法是通过检查性别和生存率来将其他标题分组到更大的篮子中以避免过度拟合  

#--Mlle  Ms  Lady  Dona都归为Miss  Mme归为Mrs
full$title[full$title == 'Mlle']        <- 'Miss' 
full$title[full$title == 'Ms']          <- 'Miss'
full$title[full$title == 'Mme']         <- 'Mrs' 
full$title[full$title == 'Lady']        <- 'Miss'
full$title[full$title == 'Dona']        <- 'Miss'


我担心用个别数据创建一个新的分类会导致过度拟合  
然而，我的想法是，将原始变量结合低于原始的变量可能会失去一些预测能力，因为他们都是军人，医生和诺贝尔人  

#--Capt  Col  Major  Dr  Rev  Don  Sir  Countess  Jonkheer归为Officer
full$title[full$title == 'Capt']        <- 'Officer'
full$title[full$title == 'Col']         <- 'Officer'
full$title[full$title == 'Major']       <- 'Officer'
full$title[full$title == 'Dr']          <- 'Officer'
full$title[full$title == 'Rev']         <- 'Officer'
full$title[full$title == 'Don']         <- 'Officer'
full$title[full$title == 'Sir']         <- 'Officer'
full$title[full$title == 'the Countess']<- 'Officer'
full$title[full$title == 'Jonkheer']    <- 'Officer'
```

#### 处理Family Groups字段
#--根据家庭成员个数离散化特征。
```r
full$FamilySize <-full$SibSp + full$Parch + 1 
full$FamilySized[full$FamilySize == 1] <- 'Single' 
full$FamilySized[full$FamilySize < 5 & full$FamilySize >= 2] <- 'Small' 
full$FamilySized[full$FamilySize >= 5] <- 'Big' 
full$FamilySized=as.factor(full$FamilySized)
```

#### 处理Tickets字段
#--工程师特征基于同一票的所有乘客。
```r
#--rep(x,times) 重复x times次   full行数个0
#--unique() 返回一个去重的向量或者数据框
ticket.unique <- rep(0, nrow(full))
tickets <- unique(full$Ticket)  #一共多少种船票

for (i in 1:length(tickets)) {
  current.ticket <- tickets[i]
  party.indexes <- which(full$Ticket == current.ticket)
  
  
  for (k in 1:length(party.indexes)) {
    ticket.unique[party.indexes[k]] <- length(party.indexes)
  }
}

full$ticket.unique <- ticket.unique
full$ticket.size[full$ticket.unique == 1]   <- 'Single'
full$ticket.size[full$ticket.unique < 5 & full$ticket.unique>= 2]   <- 'Small'
full$ticket.size[full$ticket.unique >= 5]   <- 'Big'
#--其实这个船票的分类有啥用 我真是不懂  难道是我翻译的有问题 还是这里面另有玄机？ 船票能说明什么问题
```




## 独立的变量/目标
#查看每个字段与存活率之间的关系  
#### Survival
#--变量 Survived 被标记为Bernoulli trial，其中乘客或机组成员的存活值被编码为1.在训练集的观察值中，大约38％的乘客和乘员幸存了下来。
```r
full <- full %>%
  mutate(Survived = case_when(Survived==1 ~ "Yes", 
                              Survived==0 ~ "No"))

crude_summary <- full %>%
  filter(set=="train") %>%
  select(PassengerId, Survived) %>%
  group_by(Survived) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n))

crude_survrate <- crude_summary$freq[crude_summary$Survived=="Yes"]
library(knitr)
kable(crude_summary, caption="2x2 Contingency Table on Survival.", format="markdown")
#--kable创建一个表格   Rstudio直接view()就好  
#--|Survived |   n|      freq|  
#--|:--------|---:|---------:|  
#--|No       | 549| 0.6161616|  
#--|Yes      | 342| 0.3838384|  
```

#### Exploratory data analysis(EDA)
#--探索性数据分析  
                                                  
问：什么是探索性数据分析？  
答：数据科学是数据推理，算法开发和技术的多学科融合，来解决分析复杂的问题。  

* 在统计学中，探索性数据分析（EDA）是一种分析数据集从而总结其主要特性的方法，通常配合可视化方法。 统计模型可以使用或不使用，但EDA主要是用于得到模型和假设测试任务之外的信息。  
* 相依变量与自变量的关系  
* 因变量/预测因子的关系  
* 存活率与各个变量之间的关系  

#### Pclass {-} 与存活率之间关系
```r
ggplot(full %>% filter(set=="train"), aes(Pclass, fill=Survived)) +
  geom_bar(position = "fill") +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(labels=percent) +
  ylab("Survival Rate") +
  geom_hline(yintercept=crude_survrate, col="white", lty=2, size=2) +
  ggtitle("Survival Rate by Class") + 
  theme_minimal()
```
<img src="/img/3.png"></img>


#### Sex {-} 与存活率之间关系 
```r
ggplot(full %>% filter(set=="train"), aes(Sex, fill=Survived)) +
  geom_bar(position = "fill") +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(labels=percent) +
  ylab("Survival Rate") +
  geom_hline(yintercept=crude_survrate, col="white", lty=2, size=2) +
  ggtitle("Survival Rate by Sex") + 
  theme_minimal()
```
<img src="/img/4.png"></img>


#### Age {-} 与存活率之间关系
```r
tbl_age <- full %>%
  filter(set=="train") %>%
  select(Age, Survived) %>%
  group_by(Survived) %>%
  summarise(mean.age = mean(Age, na.rm=TRUE))

ggplot(full %>% filter(set=="train"), aes(Age, fill=Survived)) +
  geom_histogram(aes(y=..density..), alpha=0.5) +
  geom_density(alpha=.2, aes(colour=Survived)) +
  geom_vline(data=tbl_age, aes(xintercept=mean.age, colour=Survived), lty=2, size=1) +
  scale_fill_brewer(palette="Set1") +
  scale_colour_brewer(palette="Set1") +
  scale_y_continuous(labels=percent) +
  ylab("Density") +
  ggtitle("Survival Rate by Age") + 
  theme_minimal()
```
<img src="/img/5.png"></img>


#### Age Groups {-} 与存活率之间关系
```r
ggplot(full %>% filter(set=="train" & !is.na(Age)), aes(`Age Group`, fill=Survived)) +
  geom_bar(position = "fill") +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(labels=percent) +
  ylab("Survival Rate") +
  geom_hline(yintercept=crude_survrate, col="white", lty=2, size=2) +
  ggtitle("Survival Rate by Age Group") + 
  theme_minimal()
```
<img src="/img/6.png"></img>


#### SibSp {-} 与存活率之间关系
```r
ggplot(full %>% filter(set=="train"), aes(SibSp, fill=Survived)) +
  geom_bar(position = "fill") +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(labels=percent) +
  ylab("Survival Rate") +
  geom_hline(yintercept=crude_survrate, col="white", lty=2, size=2) +
  ggtitle("Survival Rate by SibSp") + 
  theme_minimal()
```
<img src="/img/7.png"></img>


#### Parch {-} 与存活率之间关系
```r
ggplot(full %>% filter(set=="train"), aes(Parch, fill=Survived)) +
  geom_bar(position = "fill") +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(labels=percent) +
  ylab("Survival Rate") +
  geom_hline(yintercept=crude_survrate, col="white", lty=2, size=2) +
  ggtitle("Survival Rate by Parch") + 
  theme_minimal()
```
<img src="/img/8.png"></img>


#### Embarked {-} 与存活率之间关系
```r
ggplot(full %>% filter(set=="train"), aes(Embarked, fill=Survived)) +
  geom_bar(position = "fill") +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(labels=percent) +
  ylab("Survival Rate") +
  geom_hline(yintercept=crude_survrate, col="white", lty=2, size=2) +
  ggtitle("Survival Rate by Embarked") + 
  theme_minimal()
```
<img src="/img/9.png"></img>


#### Title {-} 与存活率之间关系
```r
ggplot(full %>% filter(set=="train") %>% na.omit, aes(title, fill=Survived)) +
  geom_bar(position="fill") +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(labels=percent) +
  ylab("Survival Rate") +
  geom_hline(yintercept=crude_survrate, col="white", lty=2, size=2) +
  ggtitle("Survival Rate by Title") + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
```
<img src="/img/10.png"></img>


#### Family {-} 与存活率之间关系
```r
ggplot(full %>% filter(set=="train") %>% na.omit, aes(`FamilySize`, fill=Survived)) +
  geom_bar(position="fill") +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(labels=percent) +
  ylab("Survival Rate") +
  geom_hline(yintercept=crude_survrate, col="white", lty=2, size=2) +
  ggtitle("Survival Rate by Family Group") + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
```
<img src="/img/11.png"></img>

                                                                                                                                          
## 变量与存活人数之间的关系

#### Pclass {-}
```r
ggplot(full %>% filter(set=="train"), aes(Pclass, fill=Survived)) +
  geom_bar(position="stack") +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(labels=comma) +
  ylab("Passengers") +
  ggtitle("Survived by Class") + 
  theme_minimal()
```
<img src="/img/12.png"></img>


#### Sex {-}
```r
ggplot(full %>% filter(set=="train"), aes(Sex, fill=Survived)) +
  geom_bar(position="stack") +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(labels=percent) +
  scale_y_continuous(labels=comma) +
  ylab("Passengers") +
  ggtitle("Survived by Sex") + 
  theme_minimal()
```
<img src="/img/13.png"></img>


#### Age {-}
```r
ggplot(full %>% filter(set=="train"), aes(Age, fill=Survived)) +
  geom_histogram(aes(y=..count..), alpha=0.5) +
  geom_vline(data=tbl_age, aes(xintercept=mean.age, colour=Survived), lty=2, size=1) +
  scale_fill_brewer(palette="Set1") +
  scale_colour_brewer(palette="Set1") +
  scale_y_continuous(labels=comma) +
  ylab("Density") +
  ggtitle("Survived by Age") + 
  theme_minimal()
```
<img src="/img/14.png"></img>


#### Age Groups {-}
```r
ggplot(full %>% filter(set=="train" & !is.na(Age)), aes(`Age Group`, fill=Survived)) +
  geom_bar(position="stack") +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(labels=comma) +
  ylab("Passengers") +
  ggtitle("Survived by Age Group") + 
  theme_minimal()
```
<img src="/img/15.png"></img>


#### SibSp {-}
```r
ggplot(full %>% filter(set=="train"), aes(SibSp, fill=Survived)) +
  geom_bar(position="stack") +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(labels=percent) +
  scale_y_continuous(labels=comma) +
  ylab("Passengers") +
  ggtitle("Survived by SibSp") + 
  theme_minimal()
```
<img src="/img/16.png"></img>


#### Parch {-}
```r
ggplot(full %>% filter(set=="train"), aes(Parch, fill=Survived)) +
  geom_bar(position="stack") +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(labels=comma) +
  ylab("Passengers") +
  ggtitle("Survived by Parch") + 
  theme_minimal()
```
<img src="/img/17.png"></img>


#### Embarked {-}
```r
ggplot(full %>% filter(set=="train"), aes(Embarked, fill=Survived)) +
  geom_bar(position="stack") +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(labels=comma) +
  ylab("Passengers") +
  ggtitle("Survived by Embarked") + 
  theme_minimal()
```
<img src="/img/18.png"></img>


#### Title {-}
```r
ggplot(full %>% filter(set=="train") %>% na.omit, aes(title, fill=Survived)) +
  geom_bar(position="stack") +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(labels=comma) +
  ylab("Passengers") +
  ggtitle("Survived by Title") + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
```
<img src="/img/19.png"></img>


#### Family {-}
```r
ggplot(full %>% filter(set=="train") %>% na.omit, aes(`FamilySize`, fill=Survived)) +
  geom_bar(position="stack") +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(labels=comma) +
  ylab("Passengers") +
  ggtitle("Survived by Family Group") + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
```
<img src="/img/20.png"></img>


## 变量之间的相互关系

#### 相关图
问：什么是Correlation Plot相关图  
答：corrplot包是相关矩阵的置信区间的图形显示。它还包含一些算法来做矩阵的重新排序。 另外，corplot擅长修改细节，包括选择颜色，文字标签，颜色标签，布局等。  
#--数字特征之间的关联方法提示冗余信息，例如* Fare *与* Pclass *。 然而，这种关系可能会由于登上作为家庭的乘客而被扭曲，其中*费用*代表家庭总成本的总和。  

```r
tbl_corr <- full %>%
  filter(set=="train") %>%
  select(-PassengerId, -SibSp, -Parch) %>%
  select_if(is.numeric) %>%
  cor(use="complete.obs") %>%
  corrplot.mixed(tl.cex=0.85)
```
<img src="/img/21.png"></img>


#### 马赛克图
问：什么是 Mosaic Plot  
答：马赛克图（也称为Marimekko图）是一种用于显示来自两个或多个定性变量的数据的图形方法。 这是spplplots的多维扩展，图形显示只有一个变量相同的信息。  

```r
tbl_mosaic <- full %>%
  filter(set=="train") %>%
  select(Survived, Pclass, Sex, AgeGroup=`Age Group`, title, Embarked, `FamilySize`) %>%
  mutate_all(as.factor)

mosaic(~Pclass+Sex+Survived, data=tbl_mosaic, shade=TRUE, legend=TRUE)
```
<img src="/img/22.png"></img>


#### 冲积图
问：什么是Alluvial Diagram冲积图  
答：冲积图是最初开发用来表示随时间变化的网络结构的一种流程图。 鉴于它们的视觉外观和对流动的重视，冲积图是以冲积扇自然形成的，这些冲积扇是由流水堆积而成的土壤自然形成的  
#--三等乘客生存的可能性最低; 然而，当* Sex *是女性时，他们的生存机会得到了改善。 令人惊讶的是，一半的幼儿和青少年遇难。 对此的合理解释可能是，这些死亡的孩子中的许多人来自大家庭，正如下面的条件推理树模型所建议的那样。  
```r
library(alluvial)

tbl_summary <- full %>%
  filter(set=="train") %>%
  group_by(Survived, Sex, Pclass, `Age Group`, title) %>%
  summarise(N = n()) %>% 
  ungroup %>%
  na.omit

alluvial(tbl_summary[, c(1:4)],
         freq=tbl_summary$N, border=NA,
         col=ifelse(tbl_summary$Survived == "Yes", "blue", "gray"),
         cex=0.65,
         ordering = list(
           order(tbl_summary$Survived, tbl_summary$Pclass==1),
           order(tbl_summary$Sex, tbl_summary$Pclass==1),
           NULL,
           NULL))
```
<img src="/img/23.png"></img>


# 机器学习算法
问：啥是机器学习  
答：机器学习是人工智能（AI）的一个应用，它提供了系统的自动学习和改进的能力，而不需要明确的程序设计。 机器学习侧重于可以访问数据并使用它自己学习的计算机程序的开发。  

#--学习过程从观察或数据开始，例如实例，直接经验或指令，以便根据我们提供的示例查找数据中的模式并在将来作出更好的决策。 主要目的是让电脑自动学习，无需人工干预或协助，并相应调整行动。  
<center><img src="https://www.mathworks.com/help/stats/machinelearningtypes.jpg"></center>

问：什么是有监督学习和无监督学习  
答：你问题真多 回去好好学学吧年轻人  
		有监督：监督学习是从标注的训练数据中推导出一个功能的数据挖掘任务。训练数据由一组训练样例组成。 在监督式学习中，每个例子都是由一个输入对象（通常是一个向量）和所需的输出值（也称为监督信号）
				监督学习算法分析训练数据并产生推断的函数，其可以用于映射新的例子。 最佳方案将允许算法正确地确定不可见实例的类标签。 这就要求学习算法以“合理”的方式从训练数据推广到看不见的情况。  
		无监督：在数据挖掘甚至数据科学领域，无监督学习任务的问题是试图找到未标记数据中的隐藏结构。 由于给学习者的例子没有标签，因此没有错误或奖励信号来评估潜在的解决方案。  
<center><img src="https://media.licdn.com/mpr/mpr/AAEAAQAAAAAAAAtrAAAAJDc2ZmQ4NDE0LTI0ODAtNDdmYi1hNDI0LThhN2M4MTFjNmYzYw.png"></center>

问：所以我们开始学习机器学习了吗  
答：嗯 这是有监督机器学习，因为我们要预测乘客的生存情况。      

问：将要学习啥算法  
答：为了便于理解，可以在这张图上查看算法统计函数的更多信息[reffer](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/)  
<center><img src="https://blogs.sas.com/content/subconsciousmusings/files/2017/04/machine-learning-cheet-sheet.png"></center>
                                                                                     
答：让我先准备一些数据集 "Pclass", "title","Sex","Embarked","FamilySized","ticket.size"  
    把这些变量 70%作为训练数据 30%作为测试数据  
问：什么是训练集 什么是测试集  
答：你自己百度去吧 懒得翻译了  

训练集：在机器学习中，训练集是用于训练模型的数据集。 在训练模型中，从训练集中挑选出特定的特征。 这些功能将被整合到模型中  
测试集：测试集是一个数据集，用于衡量模型在测试集预测方面的表现  


###Prepare and keep data set.

```r
feauter1<-full[1:891, c("Pclass", "title","Sex","Embarked","FamilySized","ticket.size")]
response <- as.factor(train$Survived)
feauter1$Survived=as.factor(train$Survived)

#--为了交叉验证，将会保留原始的训练集的20％的数据
set.seed(500)
ind=createDataPartition(feauter1$Survived,times=1,p=0.8,list=FALSE)
train_val=feauter1[ind,]
test_val=feauter1[-ind,]
```


```r
#--检查原始训练数据生存率，当前训练集和测试集数据的生存率比例
round(prop.table(table(train$Survived)*100),digits = 1)
round(prop.table(table(train_val$Survived)*100),digits = 1)
round(prop.table(table(test_val$Survived)*100),digits = 1)
```

答：让我们开始用算法训练数据吧  
问：训练完数据下一步做什么  
答：我们需要用测试的数据集验证我们训练算法的结果。  
问：我们如何衡量我们的算法性能？  
答：用[拟合度](https://en.wikipedia.org/wiki/Goodness_of_fit)，我们开始吧  
 

通过每个标签查看不同的算法  
预测分析和交叉验证  

## --------------------------------------------模型训练------------------------------------
### Decison tree {-}
#--------------------------决策树----------------------------------------------------
```r
#--随机森林比单一树更好，但是单个树很容易使用和说明
set.seed(1234)
library("rpart")
Model_DT=rpart(Survived~.,data=train_val,method="class")#构建决策树模型

library("rpart.plot")
rpart.plot(Model_DT,extra =  3,fallen.leaves = T)

```
<img src="/img/24.png"></img>

#--哇塞，检查一下画出的图形，我们单一的树模型只使用到了Title, Pclass and Ticket.size这些字段  
#--让我们预测训练数据，并检查单个树的准确性  

```r
PRE_TDT=predict(Model_DT,data=train_val,type="class")
library(caret)
confusionMatrix(PRE_TDT,train_val$Survived)  #confusionMatrix检查准确性

#####Accuracy is 0.8375  正确率0.8375
#--使用3个特征来构建树模型效果一点也不好  
#--单树中可能会过度拟合，所以我会用'10 fold techinque'方法法进行交叉验证  

set.seed(1234)
cv.10 <- createMultiFolds(train_val$Survived, k = 10, times = 10)

# Control
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10,index = cv.10)

train_val <- as.data.frame(train_val)

##Train the data  训练数据集
Model_CDT <- train(x = train_val[,-7], y = train_val[,7], method = "rpart", tuneLength = 30,trControl = ctrl)


#--检查准确性
#--单一树使用10 fold 交叉验证方法的准确率是0.8139 
#--貌似过度拟合了，之前我们的准确率是0.83
rpart.plot(Model_CDT$finalModel,extra =  3,fallen.leaves = T)
```
<img src="/img/25.png"></img>
#--嗯 模型没什么变化

```r
#--让我们使用为了测试目的而保留的数据来交叉验证准确性
PRE_VDTS=predict(Model_CDT$finalModel,newdata=test_val,type="class")
confusionMatrix(PRE_VDTS,test_val$Survived)

#--我们的训练数据和测试数据的准确性如何（0.8192）

col_names <- names(train_val)

train_val[col_names] <- lapply(train_val[col_names] , factor)
test_val[col_names] <- lapply(test_val[col_names] , factor)
```


### Random Forest {-}
#--------------------------随机森林----------------------------------------------------
```r
set.seed(1234)

library(randomForest)
rf.1 <- randomForest(x = train_val[,-7],y=train_val[,7], importance = TRUE, ntree = 1000)#构建随机森林模型
rf.1
varImpPlot(rf.1)  #随机森林测量的可变重要性的点图
```
<img src="/img/26.png"></img>

```r
#--随机森林的准确率是82.91 比决策树提高了1%
#--移除两个冗余变量 再构建一次模型看看
train_val1=train_val[,-4:-5]
test_val1=test_val[,-4:-5]


set.seed(1234)
rf.2 <- randomForest(x = train_val1[,-5],y=train_val1[,5], importance = TRUE, ntree = 1000)
rf.2
varImpPlot(rf.2)
```

<img src="/img/27.png"></img>
```r
#--你能看到变化吗  仅仅移除了两个变量  准确率就提高到了84.04
#--即使随机森林如此强大，我们也是在交叉验证之后才接受的这个模型
set.seed(2348)
cv10_1 <- createMultiFolds(train_val1[,5], k = 10, times = 10)


#--构建 caret 的训练控制对象
ctrl_1 <- trainControl(method = "repeatedcv", number = 10, repeats = 10,index = cv10_1)

set.seed(1234)
rf.5<- train(x = train_val1[,-5], y = train_val1[,5], method = "rf", tuneLength = 3,ntree = 1000, trControl =ctrl_1)
rf.5

#--交叉验证 准确率为0.8393
#--我们再预测一下测试集
pr.rf=predict(rf.5,newdata = test_val1)
confusionMatrix(pr.rf,test_val1$Survived)
#--正确率为0.8192  比我们所期望的要低
```

### lasso-ridge regression {-}
#--------------------------lasso回归----------------------------------------------------
#--（这个回归方法我也不是很熟~我是按谷歌翻译的,这一部分也没有翻译）

```r
train_val <- train_val %>%
  mutate(Survived = case_when(Survived==1 ~ "Yes", 
                              Survived==0 ~ "No"))

train_val<- as.data.frame(train_val)
train_val$title<-as.factor(train_val$title)
train_val$Embarked<-as.factor(train_val$Embarked)
train_val$ticket.size<-as.factor(train_val$ticket.size)

table(train_val$Survived)

test_val<- as.data.frame(test_val)
test_val$title<-as.factor(test_val$title)
test_val$Embarked<-as.factor(test_val$Embarked)
test_val$ticket.size<-as.factor(test_val$ticket.size)
test_val$Survived<-as.factor(test_val$Survived)


train.male = subset(train_val, train_val$Sex == "male")
train.female = subset(train_val, train_val$Sex == "female")
test.male = subset(test_val, test_val$Sex == "male")
test.female = subset(test_val, test_val$Sex == "female")


train.male$Sex = NULL

train.male$title = droplevels(train.male$title)

train.female$Sex = NULL
train.female$title = droplevels(train.female$title)

test.male$Sex = NULL
test.male$title = droplevels(test.male$title)

test.female$Sex = NULL
test.female$title = droplevels(test.female$title)

set.seed(101) 
train_ind <- sample.split(train.male$Survived, SplitRatio = .75)


# MALE

## set the seed to make your partition reproductible


cv.train.m <- train.male[train_ind, ]
cv.test.m  <- train.male[-train_ind, ]

# FEMALE
set.seed(100)

## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample.split(train.female$Survived, SplitRatio = .75)

cv.train.f <- train.male[train_ind, ]
cv.test.f  <- train.male[-train_ind, ]


x.m = data.matrix(cv.train.m[,1:5])
y.m = cv.train.m$Survived


set.seed(356)
# 10 fold cross validation
library(glmnet)
cvfit.m.ridge = cv.glmnet(x.m, y.m, 
                  family = "binomial", 
                  alpha = 0,
                  type.measure = "class")

cvfit.m.lasso = cv.glmnet(x.m, y.m, 
                  family = "binomial", 
                  alpha = 1,
                  type.measure = "class")
par(mfrow=c(1,2))
plot(cvfit.m.ridge, main = "Ridge")
plot(cvfit.m.lasso, main = "Lasso")
```
<img src="/img/29.png"></img>
```r
coef(cvfit.m.ridge, s = "lambda.min")

# Prediction on training set
PredTrain.M = predict(cvfit.m.ridge, newx=x.m, type="class")


table(cv.train.m$Survived, PredTrain.M, cv.train.m$title)

# Prediction on validation set
PredTest.M = predict(cvfit.m.ridge, newx=data.matrix(cv.test.m[,1:5]), type="class")
table(cv.test.m$Survived, PredTest.M, cv.test.m$title)


# Prediction on test set
PredTest.M = predict(cvfit.m.ridge, newx=data.matrix(test.male[,1:5]), type="class")
table(PredTest.M, test.male$title)


#female
x.f = data.matrix(cv.train.f[,1:5])
y.f = cv.train.f$Survived

set.seed(356)
cvfit.f.ridge = cv.glmnet(x.f, y.f, 
                  family = "binomial", 
                  alpha = 0,
                  type.measure = "class")
cvfit.f.lasso = cv.glmnet(x.f, y.f, 
                  family = "binomial", 
                  alpha = 1,
                  type.measure = "class")
par(mfrow=c(1,2))
plot(cvfit.f.ridge, main = "Ridge")
plot(cvfit.f.lasso, main = "Lasso")
```
<img src="/img/30.png"></img>
```r
coef(cvfit.f.ridge, s = "lambda.min")

# Ridge Model
# Prediction on training set
PredTrain.F = predict(cvfit.f.ridge, newx=x.f, type="class")
table(cv.train.f$Survived, PredTrain.F, cv.train.f$title)

confusionMatrix(cv.train.f$Survived, PredTrain.F)


# Prediction on validation set
PredTest.F = predict(cvfit.f.ridge, newx=data.matrix(cv.test.f[,1:5]), type="class")
table(cv.test.f$Survived, PredTest.F, cv.test.f$title)

confusionMatrix(cv.test.f$Survived, PredTest.F)


# Ridge Model
# Prediction on training set
PredTrain.F = predict(cvfit.f.lasso, newx=x.f, type="class")
table(cv.train.f$Survived, PredTrain.F, cv.train.f$title)

confusionMatrix(cv.train.f$Survived, PredTrain.F)

# Prediction on validation set
PredTest.F = predict(cvfit.f.lasso, newx=data.matrix(cv.test.f[,1:5]), type="class")
table(cv.test.f$Survived, PredTest.F, cv.test.f$title)

confusionMatrix(cv.test.f$Survived, PredTest.F)


# Prediction on test set
PredTest.F = predict(cvfit.f.ridge, newx=data.matrix(test.female[,1:5]), type="class")
table(PredTest.F, test.female$title)


MySubmission.F<-cbind(cv.train.m$Survived, PredTrain.M)
MySubmission.M<-cbind(cv.train.f$Survived, PredTrain.F)


MySubmission<-rbind(MySubmission.M,MySubmission.F)

colnames(MySubmission) <- c('Actual_Survived', 'predict')
MySubmission<- as.data.frame(MySubmission)

confusionMatrix(MySubmission$Actual_Survived, MySubmission$predict)

```




### Support Vector Machine - Linear Support vector Machine {-}
#-----------------------SVM-线性支持向量机-------------------------------------------
```r
###Before going to model lets tune the cost Parameter

set.seed(1274)
liner.tune=tune.svm(Survived~.,data=train_val1,kernel="linear",cost=c(0.01,0.1,0.2,0.5,0.7,1,2,3,5,10,15,20,50,100))

liner.tune

###best perforamnce when cost=3 and accuracy rate is 82.7


###Lets get a best.liner model  
best.linear=liner.tune$best.model

##Predict Survival rate using test data

best.test=predict(best.linear,newdata=test_val1,type="class")
confusionMatrix(best.test,test_val1$Survived)

###Linear model accuracy is 0.8136
```

### XGBoost {-}
#--------------------------XGBoost----------------------------------------------------
```r

library(xgboost)
library(MLmetrics)

train <- read_csv('../input/train.csv')
test  <- read_csv('../input/test.csv')

train$set <- "train"
test$set  <- "test"
test$Survived <- NA
full <- rbind(train, test)

full <- full %>%
    mutate(
      Age = ifelse(is.na(Age), mean(full$Age, na.rm=TRUE), Age),
      `Age Group` = case_when(Age < 13 ~ "Age.0012", 
                                 Age >= 13 & Age < 18 ~ "Age.1317",
                                 Age >= 18 & Age < 60 ~ "Age.1859",
                                 Age >= 60 ~ "Age.60Ov"))
                                 
full$Embarked <- replace(full$Embarked, which(is.na(full$Embarked)), 'S')


full <- full %>%
  mutate(Title = as.factor(str_sub(Name, str_locate(Name, ",")[, 1] + 2, str_locate(Name, "\\.")[, 1]- 1)))



full <- full %>%
  mutate(`Family Size`  = as.numeric(SibSp) + as.numeric(Parch) + 1,
         `Family Group` = case_when(
           `Family Size`==1 ~ "single",
           `Family Size`>1 & `Family Size` <=3 ~ "small",
           `Family Size`>= 4 ~ "large"
         ))

#--全量         
full <- full %>%
  mutate(Survived = case_when(Survived==1 ~ "Yes", 
                              Survived==0 ~ "No"))

#--出去一些变量之后的集合
full_2 <- full %>% 
  select(-Name, -Ticket, -Cabin, -set) %>%
  mutate(
    Survived = ifelse(Survived=="Yes", 1, 0)
  ) %>% 
  rename(AgeGroup=`Age Group`, FamilySize=`Family Size`, FamilyGroup=`Family Group`)


# OHE
ohe_cols <- c("Pclass", "Sex", "Embarked", "Title", "AgeGroup", "FamilyGroup")
num_cols <- setdiff(colnames(full_2), ohe_cols)

full_final <- subset(full_2, select=num_cols)

for(var in ohe_cols) {
  values <- unique(full_2[[var]])
  for(j in 1:length(values)) {
    full_final[[paste0(var,"_",values[j])]] <- (full_2[[var]] == values[j]) * 1
  }
}


submission <- TRUE

data_train <- full_final %>%
  filter(!is.na(Survived)) 

data_test  <- full_final %>% 
  filter(is.na(Survived))

set.seed(777)
ids <- sample(nrow(data_train))

# create folds for cv
n_folds <- ifelse(submission, 1, 5)

score <- data.table()
result <- data.table()



for(i in 1:n_folds) {
  
  if(submission) {
    x_train <- data_train %>% select(-PassengerId, -Survived)
    x_test  <- data_test %>% select(-PassengerId, -Survived)
    y_train <- data_train$Survived
    
  } else {
    train.ids <- ids[-seq(i, length(ids), by=n_folds)]
    test.ids  <- ids[seq(i, length(ids), by=n_folds)]
    
    x_train <- data_train %>% select(-PassengerId, -Survived)
    x_train <- x_train[train.ids,]
    
    x_test  <- data_train %>% select(-PassengerId, -Survived)
    x_test  <- x_test[test.ids,]
    
    y_train <- data_train$Survived[train.ids]
    y_test  <- data_train$Survived[test.ids]
  }
  
  x_train <- apply(x_train, 2, as.numeric)
  x_test <- apply(x_test, 2, as.numeric)
  
  if(submission) {
    nrounds <- 12
    early_stopping_round <- NULL
    dtrain <- xgb.DMatrix(data=as.matrix(x_train), label=y_train)
    dtest <- xgb.DMatrix(data=as.matrix(x_test))
    watchlist <- list(train=dtrain)
  } else {
    nrounds <- 3000
    early_stopping_round <- 100
    dtrain <- xgb.DMatrix(data=as.matrix(x_train), label=y_train)
    dtest <- xgb.DMatrix(data=as.matrix(x_test), label=y_test)
    watchlist <- list(train=dtrain, test=dtest)
  }
  
  params <- list("eta"=0.01,
                 "max_depth"=8,
                 "colsample_bytree"=0.3528,
                 "min_child_weight"=1,
                 "subsample"=1,
                 "objective"="reg:logistic",
                 "eval_metric"="auc")
  
  model_xgb <- xgb.train(params=params,
                         data=dtrain,
                         maximize=TRUE,
                         nrounds=nrounds,
                         watchlist=watchlist,
                         early_stopping_round=early_stopping_round,
                         print_every_n=2)
  
  pred <- predict(model_xgb, dtest)
  
  if(submission) {
    result <- cbind(data_test %>% select(PassengerId), Survived=round(pred, 0))
  } else {
    score <- rbind(score, 
                   data.frame(accuracy=Accuracy(round(pred, 0), y_test), best_iteration=model_xgb$best_iteration))
    temp   <- cbind(data_train[test.ids,], pred=pred)
    result <- rbind(result, temp)
  }
}

head(result)

```

### bRadial Support vector Machine {-}
#-------------------------bradial-SVM----------------------------------------------------
```r
#--让我们去非线性SVM，径向内核
set.seed(1274)

rd.poly=tune.svm(Survived~.,data=train_val1,kernel="radial",gamma=seq(0.1,5))

summary(rd.poly)
best.rd=rd.poly$best.model

#--非线性核给我们一个更好的准确性
#--让我们测试一些数据
pre.rd=predict(best.rd,newdata = test_val1)

confusionMatrix(pre.rd,test_val1$Survived)

#--非线性模型测试的数据准确率为0.81
#--也能是由于我们使用了较小的数据集用来测试
```

### Logistic Regression {-}
#--------------------------逻辑回归----------------------------------------------------
```r
contrasts(train_val1$Sex)
contrasts(train_val1$Pclass)

#--上面显示了变量如何自己编码
#--让我们运行回归模型
log.mod <- glm(Survived ~ ., family = binomial(link=logit), 
               data = train_val1)
###Check the summary
summary(log.mod)
confint(log.mod)

###Predict train data
train.probs <- predict(log.mod, data=train_val1,type =  "response")
table(train_val1$Survived,train.probs>0.5)

(395+204)/(395+204+70+45)
#--使用逻辑回归预测数据 准确性为0.83
test.probs <- predict(log.mod, newdata=test_val1,type =  "response")
table(test_val1$Survived,test.probs>0.5)

(97+47)/(97+12+21+47)
#--测试数据准确率为0.81
```

#--------------------------------------------------------------------------------------
### --机器学习算法评估

Accuracy with **Random Forest**                  - 84.03%  使用随机森林的准确率为         84.03%                                                                                                                                                          
Accuracy with **Dession trees**                  - 83.75%  使用决策树的准确率为           83.75%                                                                                                                                                                                                                                                                                                                              
Accuracy with **Radial Support vector Machine**  - 81.92%  使用支持向量机的准确率为       81.92%    
Accuracy with **lasso-ridge regression**         - 81.90%  使用lasso-ridge回归的准确率为  81.90%                                                                                                                                                                                                                                                                                                                                  
Accuracy with **Linear Support vector Machine**  - 81.36%  使用线性支持向量机的准确率为   81.36%                                                                                                                                                                
Accuracy with **Logistic Regression**            - 81.36%  使用逻辑回归的准确率为         81.36%                                                                                                                                                                    
问：哇 随机森林可以达到84.03%的准确率  
答：嗯                                                                                                                                                                                                                                                            
问：非常感谢你 James            
答：不客气  
**Please excuse any typos.**   
**Thanks for reading. If you have any feedback,suggestions I'd love to hear! .**                                                                                                      
**Please like the kernel. Your likes are my motivation. ;) **  
#巴拉巴拉巴拉  期待你的反馈  

