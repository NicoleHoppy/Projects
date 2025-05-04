## 1. Analysis of wines

The dataset we’ll be working with comes from [kaggle.com](https://www.kaggle.com/datasets/shelvigarg/wine-quality-dataset/data) and includes info on both red and white "Vinho Verde" wines — a variety from northern Portugal.

In this project, we’re going to explore how well *linear regression* can model the data.

*Linear regression* is basically a way to figure out how one variable (like wine quality) is influenced by others (like acidity, alcohol content, etc.). We try to draw a line through the data points that best captures the trend or relationship between them.

But before we dive into any analysis, let’s load up the necessary libraries so that everything runs smoothly.

```{r,message = FALSE, warning = FALSE}
library(car)
library(corrplot)
library(dplyr)
library(faraway)
library(lmtest)
library(MASS)
library(nortest)
library(RColorBrewer)
```

## 2. Dane - ich struktura oraz klasyfikacja

Let's load the winequalityN.csv file and take a quick look at the structure of the dataset.


```{r}
wine <- read.csv("C:\\Users\\Nikola\\Documents\\Nikola Chmielewska\\R\\Datasets\\winequalityN.csv")
str(wine)
```
```{r}
unique(wine$type)
```

From the command above, we can see that the dataset includes information on two wine types: red and white.

Now, let’s break down the variables included in the dataset to better understand what we’re working with:

| Name | Description | 
|:--------|:--------|
| type | Type of wine (white or red) |
| fixed.acidity |	Fixed acidity |
| volatile acidity | Volatile acidity |
| citric acid |	Citric acid |
| residual sugar | Residual sugar |
| chlorides |	Chlorides |
| free sulfur dioxide | Free sulfur dioxide |
| total sulfur dioxide | Total sulfur dioxide |
| density |	Density |
| pH – potential of hydrogen |	Acidity/basicity (pH level) |
| sulphates | Sulphates |
| alcohol | Alcohol content |
| quality |	Wine quality rating |

To get a better feel for the data, let’s look at a summary:

```{r}
summary(wine)
```

Before diving in deeper, let’s also check whether there are any missing values in the dataset.

```{r}
(sapply(wine, function(x) {sum(is.na(x))}))
```

As we can see, there are some NA values in the table. Since they represent only a small portion of the data, we’ll go ahead and drop those rows. Missing data could affect the quality of our analysis later on, so it’s better to clean it up now.

```{r}
wine_clean <- na.omit(wine) #usuwa wiersze zawierające wartości NA
```

Once we remove the rows with missing values, let’s take another look at our updated dataset.

```{r}
(sapply(wine_clean, function(x) {sum(is.na(x))}))
```

Now everything looks good — no missing data, so our dataset is ready to go!

In this dataset, the variable we’re trying to predict is quality, which represents the wine’s rating. All the other variables are independent (predictor) variables, and they’re numerical in nature. As for quality, it can actually be seen in two ways: either as a continuous numeric variable or as an ordinal categorical variable (since it reflects a quality rating).

In our analysis, we’ll first treat quality as a numeric variable when applying linear regression. But later on, when we build a proportional odds model, we’ll treat it as an ordinal categorical variable. Also, for hypothesis testing, we’ll be using a significance level of α = 0.05.

### 2.1 Splitting the Dataset

We’re going to start by splitting the dataset based on the wine type — white and red. For this analysis, we’ll keep things simple and focus only on white wine. If you want, you can run a similar analysis later for red wine by following the same steps.

```{r}
white_wine <- subset(wine_clean, type == "white")
```

Next, we’ll drop any text-based columns since we’re only interested in working with numerical data for this part of the project.

```{r}
white_numeric <- white_wine[sapply(white_wine, is.numeric)]
```

Now let’s create a few helper variables that’ll make it easier to divide the dataset into three separate subsets.

```{r}
N <- nrow(white_numeric)
I <- (1:N)
```

Time to randomly generate the indexes for the split.

```{r}
set.seed(300)
I_l <- sample.int(N, size = round(N/2)) #50% danych
I_v <- sample(setdiff(I, I_l), size = round(N/4)) #25% danych
I_t <- setdiff(setdiff(I, I_l), I_v) #25% danych
```

We’ll use those indexes to create three data subsets: a training set, a validation set, and a test set — with 50%, 25%, and 25% of the data, respectively.

```{r}
lrn <- white_numeric[I_l,] #próba ucząca
val <- white_numeric[I_v,] #próba walidacyjna
tst <- white_numeric[I_t,] #próba testowa
```

And just like that, we’ve split our data into three parts — all set for the next steps in the analysis.

### 2.2 Variable Correlation

Before jumping into building a linear regression model, let’s first take a look at the correlation heatmap for the variables in our training set.

```{r}
white_matrix <- cor(lrn)
corrplot(white_matrix, type = "lower")
```

*Correlation* tells us how two variables move together. The Pearson correlation coefficient (which ranges from -1 to 1) is a handy number that shows us the strength and direction of the relationship.

A positive value means both variables tend to increase together, while a negative value means that as one goes up, the other tends to go down. The closer the value is to either -1 or 1, the stronger the relationship. If it’s closer to 0, it usually means there’s not much of a connection.

From the plot, we can see that ‘alcohol’ and ‘density’ show the strongest relationships with wine ‘quality’ — alcohol is positively correlated, and density is negatively correlated.

It’s also worth noting that ‘density’ is strongly tied to ‘residual sugar’, and there’s also a clear link between ‘alcohol’ and ‘density’. Based on that, we’ll try building models that exclude either ‘residual sugar’ or ‘alcohol’ (or both) to see how things change.

We’ll also try out *Principal Component Analysis (PCA)* to compare this approach with standard regression.

*PCA* is a technique that transforms the original variables into a new set of uncorrelated variables (called principal components), which can make our models cleaner and sometimes more effective.

## 3. Linear Regression Models
### 3.1. Basic Models

We’re going to build several linear regression models where we try to predict wine ‘quality’ using different sets of features:

- Full model: uses all variables from the training set.
- No sugar: drops the ‘residual sugar’ variable.
- No alcohol: drops the ‘alcohol’ variable.
- No sugar & no alcohol: drops both ‘residual sugar’ and ‘alcohol’.

To make our lives easier, we’ll create a function that builds each of these models for us.

```{r}
models <<- list()
countM  = 0
add_model = function(name, model, ispca){
  models <<- append(models, list(name, model, ispca))
  countM <<- countM + 1
}
get_name  = function(index){models[[index*3-3+1]]}
get_model = function(index){models[[index*3-3+2]]}
is_pca    = function(index){models[[index*3-3+3]]}

add_model('full',                 lm(quality ~ .,                            data = lrn), FALSE)
add_model('no sugar',             lm(quality ~ . - residual.sugar,           data = lrn), FALSE)
add_model('no alcohol',           lm(quality ~ .                  - alcohol, data = lrn), FALSE)
add_model('no sugar, no alcohol', lm(quality ~ . - residual.sugar - alcohol, data = lrn), FALSE)
```

Then we’ll display a summary for each model, in the order listed above.

```{r}
for(i in 1:countM){
  print(get_name(i))
  print(summary(get_model(i)))
}
```

Just by glancing at the summaries, it’s pretty obvious that the last model (the one without both sugar and alcohol) performs the worst — especially when you look at the adjusted R-squared, which is way lower than in the other models. To double-check whether the smaller models are valid, we’ll run an ANOVA test.

As a quick reminder: adjusted R-squared tells us how well the model explains the data. Values closer to zero mean the model doesn’t do a great job.

#### 3.1.1 ANOVA

ANOVA (Analysis of Variance) is a statistical method used to compare means across different groups — basically, it helps us see if the differences between models are statistically significant. It can tell us if one model is actually better than another or if the differences are just noise.

Here’s the hypothesis setup for our test:

- H<sub>0</sub>: The smaller model is good enough.
- H<sub>1</sub>: The smaller model doesn’t fit well.

```{r}
for(i in 2:countM){
  print(paste(get_name(1), 'vs', get_name(i), sep=' '))
  print(anova(get_model(1), get_model(i)))
}
```

From the test results, at a significance level of α = 0.05, we can reject H₀ — meaning the smaller models (where we removed variables strongly correlated with density) don’t perform well enough.

#### 3.1.2. Backward Elimination + ANOVA

Now let’s try backward elimination. This method involves gradually removing the least important variables from the model, one at a time. The goal is to end up with a simpler model that still performs well.

We’ll start with the full model (the one with all variables), and begin eliminating features step by step.

```{r}
elim <- add_model('no acid', lm(quality ~ . - citric.acid, data = lrn), FALSE)
anova(get_model(1), get_model(countM))
```

```{r}
summary(get_model(countM))
```

After each removal, we’ll use ANOVA to check if the simpler model still holds up. At a significance level of *α = 0.05*, we found that we can’t reject the null hypothesis — which means the simplified models are still acceptable.

```{r}
add_model('no acid, no chlorides', lm(quality ~ . - citric.acid - chlorides, data = lrn), FALSE)
anova(get_model(1), get_model(countM))
```

```{r}
summary(get_model(countM))
```

```{r}
add_model('no acid, no chlorides, no totalsulf', lm(quality ~ . - citric.acid - chlorides - total.sulfur.dioxide, data = lrn), FALSE)
anova(get_model(1), get_model(countM))
```

```{r}
summary(get_model(countM))
```

In the next couple of models, the same thing happens. At this point, the remaining variables are statistically significant, so it doesn’t make sense to remove any more — we’d just end up weakening the model.

### 3.2. Analiza składowych głównych

W celu zbadania innego podejścia do tworzenia modelów wykorzystamy metodę składowych głównych.

```{r}
par(mfrow = c(1, 2))

pca     <- prcomp(lrn[1:11], scale. = TRUE)
pca_var <- pca$sdev^2
plot(pca_var / sum(pca_var), xlab = "Składowe główne",
 ylab = "Proporcja wariancji objaśnianej",
 type = "b")

pca_data <- data.frame(quality = lrn$quality, pca$x)

corrplot(cor(pca_data), type = "lower")
```

Jak widać na wykresie dwie zmienne, które mają najmniejsze znaczenie są poniżej poziomu istotności α = 0,05. Z wykresu obok odczytujemy także, że udało nam się utworzyć zbiór danych, w którym wszystkie zmienne (poza ‘quality’) są niezależne. Zastosujemy także tutaj metodę wstecznej eliminacji, żeby pozbyć się zmiennych, które są nieistotne w tym modelu.

### 3.3. Metoda wstecznej eliminacji i ANOVA

Na początek tworzymy model PCA, z którego będziemy wyrzucać zmienne oraz od razu sprawdzimy, czy mniejsze modele są adekwatne względem tego modelu.

```{r}
pca_base <- add_model('pca', lm(quality ~ ., data = pca_data), TRUE)
summary(get_model(countM))
```

```{r}
add_model('pca no pc6', lm(quality ~ . - PC6, data = pca_data), TRUE)
anova(get_model(pca_base), get_model(countM))
```

```{r}
summary(get_model(countM))
```

Jak widać model z wyrzuconym składnikiem ‘PC6’ jest jak najbardziej adekwatny. Spróbujmy wyrzucić jeszcze jeden składnik, tym razem ‘PC7’.

```{r}
add_model('pca no pc6, no pc7', lm(quality ~ . - PC6 - PC7, data = pca_data), TRUE)
anova(get_model(pca_base), get_model(countM))
```

```{r}
summary(get_model(countM))
```

Wynika stąd, że ten model także jest adekwatny, a przynajmniej wiemy, że nie jesteśmy w stanie odrzucić hipotezy H<sub>0</sub> na poziomie istotności α = 0,05. Dalsze wyrzucanie zmiennych nie ma sensu, ponieważ reszta zmiennych jest jak najbardziej istotna.

## 4. Diagnostyka

Przyjdziemy teraz do diagnostyki naszych nowo utworzonych modeli. Przyjrzymy się *statystyce Cooka (odległość Cooka)*, żeby wykryć obserwacje odstające, pozbyć się ich i stąd dostaniemy nowe modele. Dodatkowo policzymy procentowy udział obserwacji wpływowych. Spojrzymy też na wykresy resztowe naszych modeli i spróbujemy wyciągnąć wnioski.

Krótkie wyjaśnienie *odległość Cooka* to miara wykorzystywana w analizie regresji, służąca do wykrywania obserwacji odstających i oszacowania ich wpływu na cały model statystyczny. 

### 4.1. Obserwacje odstające

Tworzymy pomocniczą funkcję, żeby maksymalnie zautomatyzować tworzenie odległości Cooka oraz powstawanie wykresów. Pamiętamy, że za obserwację wpływową w sensie odległości Cooka uchodzą obserwacje, dla których odległość jest nie mniejsza od 1.

```{r}
cook_base = countM

cook_statistics = function(name, model, ispca){
  cook <- cooks.distance(model)
  plot(cook, xlab = "Indeksy",  ylab = paste("Odległości(", name, ")"))
  if(max(cook) >= 1){
    name <- paste('cook', name, sep = ' ')
    while(max(cook) >= 1){
      model <- update(model, subset = (cook < max(cook)))
      cook  <- cooks.distance(model)
    }
    add_model(name, model, ispca)
  }
}

layout(matrix(c(1,2,3,4,5,6,7,8,9,10), 2, 5, byrow = TRUE))

for(i in 1:cook_base){
  name  <- get_name(i)
  model <- get_model(i)
  cook_statistics(name, model, is_pca(i))
}

mtext("Odległość Cooka", side=3, outer=TRUE, line=-3)
```

Z powyższych wykresów jesteśmy w stanie wywnioskować, że modele ‘no sugar’, ‘no alcohol’ oraz ‘no sugar, no alcohol’ nie mają żadnych odstających obserwacji, ponieważ odległość Cooka jest mniejsza niż 1. Pozostałe modele posiadają obserwacje odstające, dlatego za pomocą funkcji, która została stworzona na początku tego podrozdziału, automatycznie stworzone zostały nowe modele z wyrzuceniem obserwacji odstającej. Warto dodać, że żaden z modeli nie posiadał więcej niż jednej obserwacji odstającej.

### 4.2. Obserwacje wpływowe

Zajmiemy się teraz obserwacjami wpływowymi. Wyświetlimy wykresy każdego modelu wraz z linią pokazującą, od jakiego poziomu obserwacje są obserwacjami wpływowymi. W tym celu robimy funkcję, żeby po raz kolejny zautomatyzować tworzenie wykresów, ponieważ na chwilę obecną mamy ich już 17.

```{r}
par(mfrow = c(9, 3))

leverages <- mapply(function(index){
    name  <- get_name(index)
    model <- get_model(index)
    lev   <- hat(model.matrix(model))
    p <- (index - 1) %% 3
    q <- ifelse((index - 1) %% 6 >= 3, 0, 1)
    par(fig = c(1/3 * p,1/3 + 1/3 * p, 0.5 * q, 0.5 + 0.5 * q), new = (index %% 6 != 1))
    plot(lev, xlab = "Indeksy", ylab = "Obserwacje wpływowe", main = name)
    abline(h = 2 * sum(lev) / nrow(model$model), col = 'red')
    lev
  }, 1:countM)
```

Widać, że dopiero po odrzuceniu obserwacji odstających tak powstałe modele mają bardziej przejrzyste wykresy. Jednak z nich nie jesteśmy w stanie za wiele odczytać, dlatego policzymy procentowy udział obserwacji wływowych i wyświetlimy wyniki w postaci przejrzystej tabelki.

```{r}
data.frame(names = mapply(get_name, 1:countM), 
           percentages = mapply(function(index){ 
                           model <- get_model(index)
                           lev   <- leverages[[index]]
                           paste(round(sum( ifelse(lev > 2 * sum(lev) / nrow(model$model), 1, 0)) / nrow(model$model) * 100, 2), '%')
                         }, 1:countM))
```

Widzimy różne rozłożenie procentowe obserwacji wpływowych dla różnych modeli.

### 4.3. Wykresy resztowe

Spójrzmy jeszcze tylko na wykresy resztowe wszystkich modeli.

```{r}
par(mfrow = c(ceiling(countM / 4), 4))

for(i in 1:countM){
  model <- get_model(i)
  p <- (i - 1) %% 4
  q <- ifelse((i - 1) %% 8 >= 4, 0, 1)
  par(fig = c(1/4 * p, 1/4 + 1/4 * p, 0.5 * q, 0.5 + 0.5 * q), new = (i %% 8 != 1))
  plot(model$fit, model$res, xlab="Dopasowane", ylab="Reszty", main = get_name(i))
  abline(h = 0, col = 'red')}
```

W każdym z modeli występuje pewne odchylanie reszt od wartości dopasowanych. Widzimy, że wykresy są bardzo podobne do siebie, a wszystkie bardziej odstające punkty zostały usunięte podczas tworzenia modeli stworzonych dzięki statystyce Cooka, co widać porównując odpowiednio modele podstawowe ze zmodyfikowanymi. Jedynie wykresy modeli ‘no alcohol’ oraz ‘no sugar, no alcohol’ znacząco różnią się od reszty wykresów.

## 5. Poprawność założeń modeli regresji liniowej
### 5.1. Normalność reszt

Przyjrzyjmy się najpierw wykresom.

```{r}
par(mfrow = c(ceiling(countM / 4), 4))

for(i in 1:countM){
  model <- get_model(i)
  p <- (i - 1) %% 4
  q <- ifelse((i - 1) %% 8 >= 4, 0, 1)
  par(fig = c(1/4 * p, 1/4 + 1/4 * p, 0.5 * q, 0.5 + 0.5 * q), new = (i %% 8 != 1))
  qqnorm(rstudent(model), xlab = "Teoretyczne Kwantyle", ylab = "Studentyzowane reszty", main = get_name(i))
  abline(0,1)
}
```

Z wykresów widzimy, że rozkład zmiennych resztowych nie jest zupełnie normalny, ale jest to łagodne odstępstwo od założenia normalności, ponadto próbka jest duża, więc może być zignorowane. Niemniej przeprowadźmy jeszcze test statystyczny Shapiro–Wilk. 

Ten test ma następujące hipotezy:

H<sub>0</sub>: reszty mają rozkład normalny,

H<sub>1</sub>: reszty nie mają rozkładu normalnego.

```{r}
data.frame("Nazwa modelu" = mapply(get_name, 1:countM),
           "p-value" = mapply(function(i){ shapiro.test(get_model(i)$residuals)$p.value }, 1:countM)) 
```

Z powyższego testu wynika, że powinniśmy odrzucić hipotezę H<sub>0</sub>, czyli w żadnym modelu reszty nie mają rozkładu normalnego. Rozbieżność między testami a wykresem, może wynikać z dużej próbki, dla której niektóre testy nie są adekwatne.

### 5.2. Stałość wariancji

W celu zbadania stałości wariancji zrobimy wykresy zależności zmiennych resztowych od odpowiednich zmiennych dopasowanych.

```{r}
par(mfrow = c(ceiling(countM / 4), 4))

for(i in 1:countM){
  model <- get_model(i)
  p <- (i - 1) %% 4
  q <- ifelse((i - 1) %% 8 >= 4, 0, 1)
  par(fig = c(1/4 * p, 1/4 + 1/4 * p, 0.5 * q, 0.5 + 0.5 * q), new = (i %% 8 != 1))
  plot(model$fit, model$res, xlab = "Zmienne dopasowane", ylab = "Zmienne resztowe", main = get_name(i))
  abline(h = 0, col = 'red')
}
```

Przeprowadźmy jeszcze test Goldfeld-Quandt o stałości wariancji.

H<sub>0</sub>: reszty mają stałą wariancję,

H<sub>1</sub>: reszty nie mają stałej wariancji.

```{r}
data.frame("Nazwa modelu" = mapply(get_name, 1:countM),
           "p-value" = mapply(function(i){ gqtest(get_model(i))$p.value }, 1:countM)) 
```

Widzimy, że w przypadku wszystkich modeli nie jesteśmy w stanie odrzucić hipotezy H<sub>0</sub> na poziomie istotności α = 0,05, dlatego wynika stąd, że reszty mają stałą wariancję.

### 5.3. Skorelowanie reszt

W celu sprawdzenia, czy nasze reszty są skorelowane, posłużymy się testem Durbina-Watsona:

H<sub>0</sub>: reszty nie są skorelowane,
H<sub>1</sub>: istnieje korelacja reszt.

```{r}
data.frame("Nazwa modelu" = mapply(get_name, 1:countM),
           "p-value" = mapply(function(i){ durbinWatsonTest(get_model(i))$p }, 1:countM)) 
```

Na podstawie przeprowadzonego testu statystycznego, nie mamy podstaw do odrzucenia hipotezy H<sub>0</sub> o braku korelacji reszt dla każdego modelu regresji liniowej.

## 6. Współliniowość regresorów

Aby przetestować współliniowość, posłużymy się statystyką Variance Inflation Factor.

```{r}
VIF = function(i){
  data.frame("Nazwa" = get_name(i), t(vif(get_model(i))))
}
```

Wyliczając statystykę Variance Inflation Factor, a więc prostego testu opartego na statystyce R2, który mierzy, jaka część wariancji estymatora jest powodowana przez to, że zmienna j nie jest niezależna względem pozostałych zmiennych objaśniających w modelu regresji, jesteśmy w stanie określić współliniowość dla poszczególnych zmiennych.

W każdym modelu największą miarą charakteryzują się zmienne ‘density’, ‘residual.sugar’ oraz ‘alcohol’. Możemy wywnioskować, że są to najbardziej współliniowe zmienne, natomiast nie jest to zjawisko bardzo mocno widoczne w naszym zbiorze danych. Modele bez tych parametrów są mniej współliniowe.

## 7. Miary dopasowania

Przed wybraniem najlepszego modelu, spójrzmy jeszcze na miary dopasowania modelu do danych.

```{r}
data.frame("Nazwa modelu" = mapply(get_name, 1:countM),
           "Skorygowany współczynnik determinacji"  = mapply(function(index){summary(get_model(index))$adj.r.squared}, 1:countM),
           "Estymator wariancji błędów" = mapply(function(index){summary(get_model(index))$sigma}, 1:countM),
           check.names = FALSE)
```

Widzimy, że gdybyśmy mieli wybierać najlepszy model na podstawie skorygowanego współczynnika determinacji, to wybralibyśmy model ‘cook no acid, no chlorides, no totalsulf’. Jednak wybór najlepszego modelu opieramy na wyniku resztowej sumy kwadratów.

## 8. Wybór najlepszego modelu na podstawie RSS oraz jego test

Stwórzmy funkcję, która pokaże odpowiednio: nazwę modelu; resztową sumę kwadratów pomiędzy obserwacjami empirycznymi z próby walidacyjnej, a przewidzianymi przez model regresji liniowej; odsetek poprawnych klasyfikacji; odsetek poprawnych klasyfikacji różniących się o co najwyżej jeden.

```{r}
pca_val <- as.data.frame(predict(pca, newdata = val[1:11]))
pca_tst <- as.data.frame(predict(pca, newdata = tst[1:11]))
```

```{r}
prediction_summary <- data.frame(matrix(ncol = 4, nrow = 0, dimnames = list(NULL, c("Nazwa", "RSS", "Odsetek popr","Odsetek róż"))))
```

```{r}
make_prediction = function(index, non_pca, iff_pca){
  name  <- get_name(index)
  model <- get_model(index)
  if(is_pca(index)){
    prediction <- round(predict(model, iff_pca))
  }else{
    prediction <- round(predict(model, non_pca[1:11]))
  }
  prediction_summary[nrow(prediction_summary) + 1,] <<- c(name,
    sum((prediction - non_pca[12])^2),
    sum(non_pca[12] == prediction, na.rm = TRUE) / nrow(non_pca[12]) * 100,
    sum(abs(non_pca[12] - prediction) <= 1)/ nrow(non_pca[12]) * 100)
}
```

```{r}
for(i in 1:countM){ make_prediction(i, val, pca_val) }
```

```{r}
prediction_summary
```

Poprzez utworzoną tabelę, dostrzegamy że modele o najmniejszej resztowej sumie kwadratów, a więc nasze najlepsze modele na tej podstawie, to ‘pca no pc6’ oraz ‘pca no pc6, no pc7’. Jednak model ‘pca no 6’ ma większy odsetek poprawnych klasyfikacji, dlatego to właśnie on zostanie wybrany do dalszych testów.

Przetestujmy teraz w takim razie nasz najlepszy model.

```{r}
prediction_summary = NULL
prediction_summary <- data.frame(matrix(ncol = 4, nrow = 0, dimnames = list(NULL, c("Nazwa", "RSS", "odsetek popr. kl","Odsetek kl. o 1"))))

make_prediction(9, tst, pca_tst)

prediction_summary

```

Widzimy, że nasz model nie jest wybitny, natomiast dobrze poradził sobie z próbą testową, przewidując ponad połowę poprawnych klasyfikacji. Możemy z tego wywnioskować, że jest to najlepszy utworzony przez nas model regresji liniowej, z drugiej jednak strony sama regresja liniowa jest obarczona dużym błędem w przewidywaniach.

## 9. Model proporcjonalnych szans

W celu zobrazowania zmiany zmiennej quality na zmienną jakościową utworzymy model proporcjonalnych szans:

```{r}
wine1 <- white_numeric
wine1$quality <- factor(wine1$quality)
```

Oraz podzielimy nasz zbiór na 3 podzbiory, jak wcześniej.

```{r}
lrn2 <- wine1[I_l,]
val2 <- wine1[I_v,]
tst2 <- wine1[I_t,]
```

Utworzony model wygląda następująco:

```{r}
g.plr <- polr(quality ~ ., data = lrn2)
g.plr
```

Przyjrzyjmy się teraz jego podsumowaniu.

```{r}
summary(g.plr)
```

Zastosujemy funkcję predict() do przewidywania wartości dla tego modelu.

```{r}
pr_log1 <- predict(g.plr, val2[,1:11], type="class")

val21 <- as.numeric(val2$quality)
pr_log11 <- as.numeric(pr_log1)
```

Oraz obliczmy resztową sumę kwadratów.

```{r}
sum((pr_log11-val21)^2)
```

Widzimy, że model jest przeciętny, więc ulepszymy go. Uprościmy model za pomocą funkcji step.

```{r}
o_lr = step(g.plr)
```

Utworzony model ma mniej zmiennych zależnych w celu zmniejszenia AIC, która estymuje liczbę utraconych danych przez dany model. Im mniej danych model utracił, tym lepszej jest jakości. Innymi słowy AIC określa ryzyko przeszacowania oraz niedoszacowania modelu.

Spójrzmy teraz na podsumowanie.

```{r}
summary(o_lr)
```

```{r}
anova(g.plr,o_lr)
```

Widzimy, że mniejszy model jest jak najbardziej zasadny. Policzmy resztową sumę kwadratów.

```{r}
pr_log2<-predict(o_lr, val2[,1:11],type="class")

val21<- as.numeric(val2$quality)
pr_log12 <- as.numeric(pr_log2)
sum((pr_log12-val21)^2)
```

Resztowa suma kwadratów jest większa, wiec model nie jest tak dobry jak wyjściowy, klasyfikując modele na podstawie tego kryterium, dlatego żaden z tych modeli nie jest lepszy w porównaniu z najlepszym modelem regresji liniowej.

## 10. Podsumowanie

W projekcie utworzone zostało 17 modeli regresji liniowych oraz 2 modele proporcjonalnych szans.

Wśród wszystkich modeli najmniejszą wartość resztowej sumy kwadratów osiągnął model regresji liniowej, który został ostatecznie przetestowany dla zbioru testowego. Na jego podstawie określiliśmy, że model regresji liniowej jest całkiem dokładny, ponieważ próba testowa dała nam bardzo dobry wynik.

Można oczywiście tworzyć następne modele, wyrzucać kolejne zmienne odstające i je analizować, jednak my postanowiliśmy ograniczyć się tylko do tych kilku modeli. Natomiast widać spory problem w tym, że regresja liniowa nie jest idealnym modelem dla naszego zbioru danych.
