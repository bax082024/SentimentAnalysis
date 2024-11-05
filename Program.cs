using System;
using System.Reflection.Emit;
using Microsoft.ML;
using Microsoft.ML.Data;

class Program
{
    static void Main(string[] args)
    {
        MLContext mlContext = new MLContext();

        var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text));

        var trainer = mlContext.BinaryClassification.Trainers.FastTree(
            labelColumnName: "Label", 
            featureColumnName: "Features",
            numberOfLeaves: 100,
            learningRate: 0.2,
            numberOfTrees: 200);

        var trainingPipeline = dataProcessPipeline.Append(trainer);

        var model = trainingPipeline.Fit(LoadData(mlContext));

        // Cross-validation
        var cvResults = mlContext.BinaryClassification.CrossValidate(data: LoadData(mlContext), estimator: trainingPipeline, numberOfFolds: 5);
        var avgAccuracy = cvResults.Average(r => r.Metrics.Accuracy);
        Console.WriteLine($"Cross-validated Accuracy: {avgAccuracy:P2}");



        

        

        var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

        var testSentiments = new[]
        {
          new SentimentData { Text = "I love this product!"},
          new SentimentData { Text = "This is the worst experience i have ever had"},
          new SentimentData { Text = "Im not sure how i feel about this"}
        };

        Console.WriteLine("Predictions:");
        foreach (var sentiment in testSentiments)
        {
          var result = predictionEngine.Predict(sentiment);
          Console.WriteLine($"Text: {sentiment.Text}");
          Console.WriteLine($"Prediction: {(result.Prediction ? "Positive" : "Negative")}, Probability: {result.Probability:P2}");
        }
    }

   


    static IDataView LoadData(MLContext mlContext)
    {
        var trainingData = new[]
        {
            new SentimentData { Text = "This product is great!", Label = true },
            new SentimentData { Text = "I did not like this product at all.", Label = false },
            new SentimentData { Text = "This is the best thing I have ever bought.", Label = true },
            new SentimentData { Text = "Absolutely fantastic experience!", Label = true },
            new SentimentData { Text = "Highly recommend this to everyone.", Label = true },
            new SentimentData { Text = "Worst product I have ever bought.", Label = false },
            new SentimentData { Text = "Terrible, would not buy again.", Label = false },
            new SentimentData { Text = "Would not recommend to anyone.", Label = false },
            new SentimentData { Text = "I'm extremely satisfied with this purchase.", Label = true },
            new SentimentData { Text = "Very disappointing experience.", Label = false },
            new SentimentData { Text = "This exceeded my expectations.", Label = true },
            new SentimentData { Text = "Awful, not worth the money.", Label = false },
            new SentimentData { Text = "Best thing I've ever owned.", Label = true },
            new SentimentData { Text = "Worst purchase ever", Label = false },
            new SentimentData { Text = "best experience ever!", Label = true},
            new SentimentData { Text = "im not sure how i feel about this", Label = false},
            new SentimentData { Text = "This was exactly what i needed !",  Label = true },
            new SentimentData { Text = "this is horrible!", Label = false},
            new SentimentData { Text = "this product is bad", Label = false},
            new SentimentData { Text = "i really dont know what to do", Label = false},
            new SentimentData { Text = "there is nothing good about this!", Label = false},
            new SentimentData { Text = "I absolutely love this product!", Label = true },
            new SentimentData { Text = "I would definitely buy this again.", Label = true },
            new SentimentData { Text = "This is the best purchase I've made.", Label = true },
            new SentimentData { Text = "I am so disappointed with this.", Label = false },
            new SentimentData { Text = "This product didn’t meet my expectations.", Label = false },
            new SentimentData { Text = "It's okay, not great but not bad either.", Label = false },
            new SentimentData { Text = "An average experience overall.", Label = false },
            new SentimentData { Text = "Highly satisfied with the results!", Label = true },
            new SentimentData { Text = "I wouldn’t recommend this product to anyone.", Label = false },
            new SentimentData { Text = "This was a complete waste of money.", Label = false },
            new SentimentData { Text = "Amazing product, works perfectly!", Label = true },
            new SentimentData { Text = "I regret buying this.", Label = false },
            new SentimentData { Text = "This item exceeded my expectations!", Label = true },
            new SentimentData { Text = "I don’t see the hype around this.", Label = false },
            new SentimentData { Text = "It was fine, nothing special.", Label = false },
            new SentimentData { Text = "One of the best things I've bought recently.", Label = true },
            new SentimentData { Text = "The quality is outstanding.", Label = true },
            new SentimentData { Text = "This product is very low quality.", Label = false },
            new SentimentData { Text = "I am extremely unhappy with this purchase.", Label = false },
            new SentimentData { Text = "Works exactly as advertised.", Label = true },
            new SentimentData { Text = "Not worth the price.", Label = false },
            new SentimentData { Text = "I'm not impressed with this.", Label = false },
            new SentimentData { Text = "This product is a total disaster.", Label = false },
            new SentimentData { Text = "I am so impressed with this product!", Label = true },
            new SentimentData { Text = "Completely useless and a waste of money.", Label = false },
            new SentimentData { Text = "Best decision ever to buy this!", Label = true },
            new SentimentData { Text = "This item is so poorly made.", Label = false },
            new SentimentData { Text = "Absolutely perfect for my needs.", Label = true },
            new SentimentData { Text = "I would give this zero stars if I could.", Label = false },
            new SentimentData { Text = "The quality is amazing, highly recommended.", Label = true },
            new SentimentData { Text = "Such a disappointment, don't buy it.", Label = false },
            new SentimentData { Text = "I'm beyond satisfied with this purchase.", Label = true },
            new SentimentData { Text = "It broke after just one use, terrible quality.", Label = false },
            new SentimentData { Text = "This is by far the best item I've ever purchased.", Label = true },
            new SentimentData { Text = "Completely not what I expected, very let down.", Label = false },
            new SentimentData { Text = "I am very happy with how this works.", Label = true },
            new SentimentData { Text = "This product feels cheap and flimsy.", Label = false },
            new SentimentData { Text = "Exactly what I was looking for!", Label = true },
            new SentimentData { Text = "I regret buying this.", Label = false },
            new SentimentData { Text = "Fantastic, I love it!", Label = true },
            new SentimentData { Text = "This was a complete waste.", Label = false },
            new SentimentData { Text = "Couldn't be more pleased with my purchase.", Label = true },
            new SentimentData { Text = "Worst experience ever.", Label = false },
            new SentimentData { Text = "I am thrilled with how this turned out!", Label = true },
            new SentimentData { Text = "Cheap and unreliable.", Label = false },
            new SentimentData { Text = "Best thing I've spent money on.", Label = true },
            new SentimentData { Text = "Not worth a single penny.", Label = false },
            new SentimentData { Text = "So happy I decided to buy this.", Label = true },
            new SentimentData { Text = "A total failure.", Label = false },
            new SentimentData { Text = "Couldn't be happier with this purchase!", Label = true },
            new SentimentData { Text = "Very disappointing.", Label = false },
            new SentimentData { Text = "I would recommend this to everyone.", Label = true },
            new SentimentData { Text = "Complete letdown.", Label = false },
            new SentimentData { Text = "Excellent value for money.", Label = true },
            new SentimentData { Text = "Do not waste your money on this.", Label = false },
            new SentimentData { Text = "Very pleased with the results.", Label = true },
            new SentimentData { Text = "It's awful.", Label = false },
            new SentimentData { Text = "Works exactly as described.", Label = true },
            new SentimentData { Text = "Extremely disappointing.", Label = false },
            new SentimentData { Text = "Couldn't have asked for more.", Label = true },
            new SentimentData { Text = "Absolutely horrible experience.", Label = false },
            new SentimentData { Text = "I am ecstatic about this product!", Label = true },
            new SentimentData { Text = "It fell apart quickly.", Label = false },
            new SentimentData { Text = "Very reliable and durable.", Label = true },
            new SentimentData { Text = "Terrible customer service and product.", Label = false },
            new SentimentData { Text = "Five stars from me!", Label = true },
            new SentimentData { Text = "Did not meet my expectations.", Label = false },
            new SentimentData { Text = "Perfect for what I needed.", Label = true },
            new SentimentData { Text = "Not happy with this at all.", Label = false },
            new SentimentData { Text = "This exceeded my expectations in every way.", Label = true },
            new SentimentData { Text = "Very poor craftsmanship.", Label = false },
            new SentimentData { Text = "Highly recommend to anyone.", Label = true },
            new SentimentData { Text = "Terrible choice, regret buying it.", Label = false },
            new SentimentData { Text = "Exceptional quality, very pleased.", Label = true },
            new SentimentData { Text = "Waste of resources.", Label = false },
            new SentimentData { Text = "I am absolutely loving this.", Label = true },
            new SentimentData { Text = "Totally unreliable.", Label = false },
            new SentimentData { Text = "It works like a charm.", Label = true },
            new SentimentData { Text = "Subpar experience overall.", Label = false },
            new SentimentData { Text = "One of my best purchases.", Label = true },
            new SentimentData { Text = "Would not buy this again.", Label = false },
            new SentimentData { Text = "I feel very satisfied with this purchase.", Label = true },
            new SentimentData { Text = "An utter disappointment.", Label = false },
            new SentimentData { Text = "This product is flawless.", Label = true },
            new SentimentData { Text = "Regret spending money on this.", Label = false },
            new SentimentData { Text = "Outstanding product and value.", Label = true },
            new SentimentData { Text = "Doesn't live up to the hype.", Label = false },
            new SentimentData { Text = "I am extremely pleased with this purchase.", Label = true },
            new SentimentData { Text = "Completely useless.", Label = false },
            new SentimentData { Text = "I'm very impressed with this item.", Label = true },
            new SentimentData { Text = "This is not worth the hassle.", Label = false },
            new SentimentData { Text = "It’s exactly what I hoped for.", Label = true },
            new SentimentData { Text = "Poor performance and build quality.", Label = false },
            new SentimentData { Text = "I'm very impressed with this purchase.", Label = true },
            new SentimentData { Text = "I don't see why anyone would buy this.", Label = false },
            new SentimentData { Text = "Amazing, it’s better than expected.", Label = true },
            new SentimentData { Text = "It failed to work as promised.", Label = false },
            new SentimentData { Text = "I'm beyond happy with this purchase.", Label = true },
            new SentimentData { Text = "Such a waste, would not buy again.", Label = false },
            new SentimentData { Text = "It’s great, I love it!", Label = true },
            new SentimentData { Text = "Highly disappointing purchase.", Label = false },
            new SentimentData { Text = "An amazing find!", Label = true },
            new SentimentData { Text = "Absolutely not worth it.", Label = false },
            new SentimentData { Text = "I’m thrilled with my decision.", Label = true },
            new SentimentData { Text = "Quality is below average.", Label = false },
            new SentimentData { Text = "Perfect in every way.", Label = true },
            new SentimentData { Text = "Disappointed in every aspect.", Label = false },
            new SentimentData { Text = "Worth every penny.", Label = true },
            new SentimentData { Text = "It’s defective and useless.", Label = false },
            new SentimentData { Text = "I am delighted with this.", Label = true },
            new SentimentData { Text = "The worst decision I've made.", Label = false },
            new SentimentData { Text = "It’s perfect, just as described.", Label = true },
            new SentimentData { Text = "I regret ever buying this.", Label = false },
            new SentimentData { Text = "Fantastic, it’s just what I wanted!", Label = true },
            new SentimentData { Text = "This is a terrible item.", Label = false },
            new SentimentData { Text = "It’s phenomenal!", Label = true },
            new SentimentData { Text = "Waste of money and time.", Label = false },
            new SentimentData { Text = "Very happy with the quality.", Label = true },
            new SentimentData { Text = "I had a bad experience with this.", Label = false },
            new SentimentData { Text = "It’s everything I hoped for.", Label = true },
            new SentimentData { Text = "Regretful purchase.", Label = false },
            new SentimentData { Text = "Impressive and works great.", Label = true },
            new SentimentData { Text = "Extremely poor quality.", Label = false },
            new SentimentData { Text = "I absolutely recommend this.", Label = true },
            new SentimentData { Text = "Very cheaply made.", Label = false },
            new SentimentData { Text = "I couldn't be happier.", Label = true },
            new SentimentData { Text = "One of my worst buys.", Label = false },
            new SentimentData { Text = "Very useful and handy!", Label = true },
            new SentimentData { Text = "Exceeded my expectations in every way.", Label = true },
            new SentimentData { Text = "Terrible experience, would not recommend.", Label = false },
            new SentimentData { Text = "Incredible product! Highly recommended.", Label = true },
            new SentimentData { Text = "This item did not work as expected.", Label = false },
            new SentimentData { Text = "Such a wonderful experience using this.", Label = true },
            new SentimentData { Text = "Terrible quality, broke after a day.", Label = false },
            new SentimentData { Text = "The product was okay, but nothing special.", Label = false },
            new SentimentData { Text = "Fantastic performance, exceeded my expectations.", Label = true },
            new SentimentData { Text = "Awful, I would never buy this again.", Label = false },
            new SentimentData { Text = "Extremely satisfied with my purchase!", Label = true },
            new SentimentData { Text = "Not worth the money at all.", Label = false },
            new SentimentData { Text = "Best product I've used in a long time.", Label = true },
            new SentimentData { Text = "This is simply amazing!", Label = true },
            new SentimentData { Text = "Broke after a week, very disappointed.", Label = false },
            new SentimentData { Text = "Perfect, just what I was looking for.", Label = true },
            new SentimentData { Text = "This was a huge disappointment.", Label = false },
            new SentimentData { Text = "Five stars! Highly recommend.", Label = true },
            new SentimentData { Text = "Two stars, did not live up to the hype.", Label = false },
            new SentimentData { Text = "Exceeded all my expectations.", Label = true },
            new SentimentData { Text = "Would not recommend this to anyone.", Label = false },
            new SentimentData { Text = "Simply the best product out there.", Label = true },
            new SentimentData { Text = "Regret buying this, total waste.", Label = false },
            new SentimentData { Text = "Impressive build quality and features.", Label = true },
            new SentimentData { Text = "Poorly made, feels cheap.", Label = false },
            new SentimentData { Text = "Absolutely love it!", Label = true },
            new SentimentData { Text = "Not what I expected, disappointed.", Label = false },
            new SentimentData { Text = "A fantastic product that I enjoy using.", Label = true },
            new SentimentData { Text = "Unreliable and disappointing.", Label = false },
            new SentimentData { Text = "I couldn't be happier with this purchase.", Label = true },
            new SentimentData { Text = "Did not meet my expectations.", Label = false },
            new SentimentData { Text = "Great for the price!", Label = true },
            new SentimentData { Text = "Would not purchase again.", Label = false },
            new SentimentData { Text = "Very pleased with the quality.", Label = true },
            new SentimentData { Text = "This is unacceptable, waste of money.", Label = false },
            new SentimentData { Text = "It works like a charm!", Label = true },
            new SentimentData { Text = "Feels very low quality.", Label = false },
            new SentimentData { Text = "Best decision I’ve made in a while.", Label = true },
            new SentimentData { Text = "Does not work as advertised.", Label = false },
            new SentimentData { Text = "Worth every penny!", Label = true },
            new SentimentData { Text = "This is a joke, it broke immediately.", Label = false },
            new SentimentData { Text = "I am very happy with this purchase.", Label = true },
            new SentimentData { Text = "Not good, poor experience.", Label = false },
            new SentimentData { Text = "Excellent quality and value.", Label = true },
            new SentimentData { Text = "Worst purchase I've made in a while.", Label = false },
            new SentimentData { Text = "This product changed my life!", Label = true },
            new SentimentData { Text = "Wouldn't buy this again.", Label = false },
            new SentimentData { Text = "Top-notch product, very impressed.", Label = true },
            new SentimentData { Text = "Terrible design, very disappointed.", Label = false },
            new SentimentData { Text = "I am thrilled with this product.", Label = true },
            new SentimentData { Text = "Very dissatisfied, returning it.", Label = false },
            new SentimentData { Text = "Perfect for my needs!", Label = true },
            new SentimentData { Text = "Waste of money, do not buy.", Label = false },
            new SentimentData { Text = "I can't believe how good this is!", Label = true },
            new SentimentData { Text = "Very poor performance, regret it.", Label = false },
            new SentimentData { Text = "Incredible, exceeded my expectations.", Label = true },
            new SentimentData { Text = "Absolutely terrible experience.", Label = false },
            new SentimentData { Text = "A great value for the quality.", Label = true },
            new SentimentData { Text = "Not worth the high price.", Label = false },
            new SentimentData { Text = "Very satisfied with this product.", Label = true },
            new SentimentData { Text = "I expected much more.", Label = false },
            new SentimentData { Text = "High quality and reliable.", Label = true },
            new SentimentData { Text = "This is complete garbage.", Label = false },
            new SentimentData { Text = "Absolutely flawless!", Label = true },
            new SentimentData { Text = "Disappointed in the quality.", Label = false },
            new SentimentData { Text = "This is exactly what I needed.", Label = true },
            new SentimentData { Text = "Did not work as promised.", Label = false },
            new SentimentData { Text = "Best purchase I’ve made in a long time.", Label = true },
            new SentimentData { Text = "This is cheaply made.", Label = false },
            new SentimentData { Text = "I'm very pleased with this item.", Label = true },
            new SentimentData { Text = "Fell apart almost immediately.", Label = false },
            new SentimentData { Text = "This is amazing, highly recommend.", Label = true },
            new SentimentData { Text = "The worst, absolute waste.", Label = false },
            new SentimentData { Text = "I really enjoy using this.", Label = true },
            new SentimentData { Text = "Very disappointed, would not buy.", Label = false },
            new SentimentData { Text = "This product is outstanding.", Label = true },
            new SentimentData { Text = "Not worth buying.", Label = false },
            new SentimentData { Text = "The best investment I made.", Label = true },
            new SentimentData { Text = "Awful quality, save your money.", Label = false },
            new SentimentData { Text = "This product is worth every cent.", Label = true },
            new SentimentData { Text = "Broke right after the return period.", Label = false },
            new SentimentData { Text = "Very happy with the results.", Label = true },
            new SentimentData { Text = "Terrible, doesn't work at all.", Label = false },
            new SentimentData { Text = "Excellent, would buy again.", Label = true },
            new SentimentData { Text = "I am not satisfied with this.", Label = false },
            new SentimentData { Text = "Great experience, highly recommended.", Label = true },
            new SentimentData { Text = "This was a complete letdown.", Label = false },
            new SentimentData { Text = "Very well made and durable.", Label = true },
            new SentimentData { Text = "Would not buy again, poor quality.", Label = false }
        };

        var positives = trainingData.Count(data => data.Label == true);
        var negatives = trainingData.Count(data => data.Label == false);

        Console.WriteLine($"Positive samples: {positives}, Negative samples: {negatives}");


        return mlContext.Data.LoadFromEnumerable(trainingData);
    }
}

public class SentimentData
{
    [LoadColumn(0)] 
    public string? Text { get; set; }

    [LoadColumn(1)] 
    public bool Label { get; set; } 
}

public class SentimentPrediction : SentimentData
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
    
    public float Probability { get; set; }
    
    public float Score { get; set; }
}
