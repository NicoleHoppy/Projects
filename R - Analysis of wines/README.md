## 1. Analysis of wines

The dataset used for the following analysis comes from kaggle.com. It contains data of both red and white variants of "Vinho Verde" wine, originating from northern Portugal.

The purpose of this analysis is to apply linear regression.

*Linear regression* is a statistical method used to examine the relationship between one dependent variable and one or more independent variables. It involves attempting to fit a line to the data in order to understand the nature of the relationship between them.

First, we will load the necessary libraries so that the functions used in the project work correctly.

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

Załadujmy plik winequalityN.csv oraz prześledźmy jego strukturę danych.

Załadujmy plik winequalityN.csv oraz prześledźmy jego strukturę danych.

```{r}
wine <- read.csv("C:\\Users\\Nikola\\Documents\\Nikola Chmielewska\\R\\Datasets\\winequalityN.csv")
str(wine)
```
```{r}
unique(wine$type)
```

Na podstawie powyższej komendy widzimy, że zbiór danych zawiera również informacje o wariancie wina: czerwonym i białym.

Dokonajmy jeszcze objaśnienia zmiennych występujących w zbiorze danych.

| Nazwa | Opis | 
|:--------|:--------|
| type | rodzaj wina (białe, czerwone) |
| fixed.acidity | stała kwasowość |
| volatile acidity | zmienna kwasowość |
| citric acid | kwas cytrynowy |
| residual sugar | resztowy cukier |
| chlorides |	chlorki |
| free sulfur dioxide | 	wolny dwutlenek siarki |
| total sulfur dioxide | 	całkowity dwutlenek siarki |
| density |	gęstość |
| pH – potential of hydrogen | współczynnik kwasowości/zasadowości pH |
| sulphates | siarczyny |
| alcohol | alkohol |
| quality |	jakość |

Oraz, aby lepiej zrozumieć dane, zobaczmy ich podsumowanie:

```{r}
summary(wine)
```

Sprawdźmy jeszcze, czy w zbiorze wszystkie dane są podane:

```{r}
(sapply(wine, function(x) {sum(is.na(x))}))
```
Jak widać powyżej, nasza tabela zawiera wartości NA. Usuniemy zatem wiersze, które nie posiadają danych, ponieważ w kontekście całego zbioru danych jest to marginalna ilość. Robimy to, ponieważ brak danych może obniżyć jakość późniejszej analizy danych.

```{r}
wine_clean <- na.omit(wine) #usuwa wiersze zawierające wartości NA
```

Wyświetlmy jeszcze raz nasz nowy zbiór danych z usuniętymi wartościami NA.

```{r}
(sapply(wine_clean, function(x) {sum(is.na(x))}))
```

Jak widzimy, w tej chwili żadna zmienna nie jest obarczona brakiem danych, więc nasz zbiór danych jest gotowy do obróbki.

Jednocześnie w powyższym zestawie danych zmienną zależną jest atrybut quality, a więc ocena jakości wina. Pozostałe zmienne są zmiennymi niezależnymi, ponieważ wpływają one na końcową ocenę wina; są to również zmienne ilościowe. Natomiast zmienną objaśnianą ‘quality’ można potraktować zarówno jako zmienną ilościową oraz jako zmienną nominalną o uporządkowanych kategoriach (jakościową). 

W niniejszej pracy zmienną 'quality' potraktujemy jako zmienną ilościową, rozważając modele regresji liniowej. Jednak podczas tworzenia modelu proporcjonalnych szans, zmienną ‘quality’ potraktujemy jako zmienną jakościową. Warto w tym miejscu zaznaczyć, że w celu weryfikacji hipotez będziemy zakładać poziom istotności *α = 0,05*.

### 2.1 Podział zbioru danych

Podzielimy nasz zbiór danych na podstawie wariantu wina - białe i czerwone. W niniejszej analizie skupimy się jedynie na przeanalizowaniu danych dot. białego wina. Dla czerwonego wariantu wina można zrobić osobną analizę, przechodząc analogicznie przez kroki, jakie wykonaliśmy dla białego wina.

```{r}
white_wine <- subset(wine_clean, type == "white")
```

Dodatkowo usuniemy kolumny tekstowe, ponieważ tylko kolumny z danymi numerycznymi będą nam potrzebne do dalszej analizy.

```{r}
white_numeric <- white_wine[sapply(white_wine, is.numeric)]
```

Stworzymy teraz zmienne pomocnicze, które pomogą nam podzielić zbiór danych na 3 podzbiory.

```{r}
N <- nrow(white_numeric)
I <- (1:N)
```

Teraz losujemy indeksy.

```{r}
set.seed(300)
I_l <- sample.int(N, size = round(N/2)) #50% danych
I_v <- sample(setdiff(I, I_l), size = round(N/4)) #25% danych
I_t <- setdiff(setdiff(I, I_l), I_v) #25% danych
```

Przypiszmy im konkretne dane, dzieląc nasz wyjściowy zbiór danych na 3 podzbiory: próbę uczącą, walidacyjną oraz testową o udziale procentowym, odpowiednio, 50%, 25% oraz 25% danych wyjściowych.

```{r}
lrn <- white_numeric[I_l,] #próba ucząca
val <- white_numeric[I_v,] #próba walidacyjna
tst <- white_numeric[I_t,] #próba testowa
```

W ten sposób podzieliliśmy zbiór danych na trzy podzbiory, które będą nam pomocne do dalszej analizy.

### 2.2 Korelacja zmiennych

Zanim utworzymy model regresji liniowej, zobaczmy jak prezentuje się wykres korelacji poszczególnych zmiennych w naszej próbie uczącej.

```{r}
white_matrix <- cor(lrn)
corrplot(white_matrix, type = "lower")
```
*Korelacja* określa wzajemne powiązanie między wybranymi zmiennymi. Wyrazem liczbowym korelacji jest współczynnik korelacji (R Pearsona) zawierający się w przedziale [-1,1].

Wartość dodatnia korelacji oznacza, że wraz ze wzrostem jednej cechy, następuje wzrost drugiej. Wartość ujemna natomiast określa, że wraz ze wzrostem jednej, następuje spadek drugiej cechy. W takim razie wartości najbliżej krańców przedziału wskazują największe korelacje. Z drugiej strony wartości bliżej zera wskazują na brak powiązania zmiennych.

Widzimy, że najbardziej skorelowaną zmienną ze zmienną ‘quality’, jest ‘alcohol’ oraz ‘density’, gdzie są to zmienne odpowiednio skorelowane dodatnio oraz ujemnie.

Warto też zauważyć, że mocną wzajemną korelację wykazują zmienne ‘density’ i ‘residual.sugar’ oraz zmienne ‘alcohol’ i ‘density’. Mając na uwadze te obserwacje, stworzymy modele bez ‘residual.sugar’ i ‘alcohol’, żeby zobaczyć, jakie wyniki wtedy dostaniemy.

Zastosujemy również *metodę głównych składowych (PCA)*, żeby porównać oba podejścia.

*Metoda głównych składowych (PCA)* polega na transformacji oryginalnego zbioru zmiennych na nowy zestaw nieskorelowanych ze sobą zmiennych, zwanych składowymi głównymi.

## 3. Modele regresji liniowej
### 3.1. Modele podstawowe

Stworzymy modele regresji liniowej zmiennej ‘quality’ względem, odpowiednio:

- wszystkich zmiennych występujących w próbce uczącej (full),
- wszystkich zmiennych występujących w próbce uczącej z wyrzuceniem zmiennej ‘residual.sugar’ (no sugar),
- wszystkich zmiennych występujących w próbce uczącej z wyrzuceniem zmiennej ‘alcohol’ (no alcohol),
- wszystkich zmiennych występujących w próbce uczącej z wyrzuceniem zmiennych ‘residual.sugar’ oraz ‘alcohol’ (no sugar, no alcohol).

Tworzymy w tym celu funkcję, żeby zautomatyzować ten proces.

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

Poniżej wyświetlamy podsumowanie każdego modelu w podanej wyżej kolejności.

```{r}
for(i in 1:countM){
  print(get_name(i))
  print(summary(get_model(i)))
}
```

Już na pierwszy rzut oka widać, że ostatni model nie jest w żaden sposób konkurencyjny względem reszty modeli, biorąc pod uwagę chociażby *skorygowany współczynnik determinacji*, który jest nieporównywalnie mniejszy w stosunku do reszty modeli. Jednak w celu zweryfikowania, czy modele mniejsze są adekwatne, posłużymy się testem ANOVA.

Jak chodzi o interpretację *skorygowanego współczynnika determinacji*, to przyjmuje się, że wartości bliżej zera oznacza model, który nie ma wartości predykcyjnej.

#### 3.1.1 ANOVA

*ANOVA*, Analiza wariancji to rodzina modeli statystycznych i powiązanych z nimi metod estymacji i wnioskowania wykorzystywanych do analizy różnic pomiędzy średnimi w różnych populacjach, np. w zależności od jednego lub wielu działających równoczeeśnie czynników. W najprostszej formie ANOVA stanowi test statystyczny sprawdzający czy dwie lub więcej średnich w populacjach jest sobie równych.

Nasz test statystyczny ma postać:

- H<sub>0</sub>: mniejszy model jest adekwatny,
- H<sub>1</sub>: mniejszy model nie jest adekwatny.

```{r}
for(i in 2:countM){
  print(paste(get_name(1), 'vs', get_name(i), sep=' '))
  print(anova(get_model(1), get_model(i)))
}
```

Widzimy stąd, że na poziomie istotności α=0.05 jesteśmy w stanie odrzucić hipotezę H<sub>0</sub>. Oznacza to, że żaden z modeli powstałych przez wyrzucenie zmiennych mocno skorelowanych z density nie jest adekwatny.

#### 3.1.2. Metoda wstecznej eliminacji i ANOVA

W tym podrozdziale zastosujemy *metodę wstecznej eliminacji*. Polega ona na kolejnym wyrzucaniu zmiennych, które są najmniej istotne w modelu. Dzięki temu dostaniemy nowe modele. które posłużą nam do dalszej analizy.

Zaznaczmy, że będziemy wyrzucać zmienne z pierwszego modelu, czyli z modelu regresji liniowej zmiennej ‘quality’ względem wszystkich zmiennych występujących w próbce uczącej.

```{r}
elim <- add_model('no acid', lm(quality ~ . - citric.acid, data = lrn), FALSE)
anova(get_model(1), get_model(countM))
```

```{r}
summary(get_model(countM))
```

Przed wyświetleniem podsumowania zastosowaliśmy od razu test ANOVA, żeby zweryfikować, czy mniejszy model jest adekwatny. Na poziomie istotności α = 0,05 nie jesteśmy w stanie odrzucić hipotezy H<sub>0</sub>.

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

Podobna sytuacja występuje w dwóch kolejnych modelach. Nie jesteśmy w stanie już dalej odrzucać zmiennych, ponieważ na poziomie istotności α = 0.05 kolejne zmienne są istotne, więc usuwanie ich nie jest sensowne.

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
