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
            numberOfLeaves: 60,               // Increased leaves for complexity
            learningRate: 0.05,                 // Lower learning rate for generalization
            numberOfTrees: 120,                // Increased trees for better accuracy
            minimumExampleCountPerLeaf: 10     // Ensures a minimum of 20 samples per leaf
        );


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
          new SentimentData { Text = "This is perfect!"}
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
        // Positive samples
        new SentimentData { Text = "This product is great!", Label = true },
        new SentimentData { Text = "Absolutely fantastic experience!", Label = true },
        new SentimentData { Text = "Absolutely thrilled with this purchase!", Label = true },
        new SentimentData { Text = "Exceeded my expectations in every way!", Label = true },
        new SentimentData { Text = "Best investment I've made recently.", Label = true },
        new SentimentData { Text = "I am so happy with this!", Label = true },
        new SentimentData { Text = "Fantastic, I would buy this again in a heartbeat.", Label = true },
        new SentimentData { Text = "Incredible value for the quality.", Label = true },
        new SentimentData { Text = "Couldn’t be happier with how it turned out.", Label = true },
        new SentimentData { Text = "Perfect for what I needed!", Label = true },
        new SentimentData { Text = "Exactly as described, very pleased.", Label = true },
        new SentimentData { Text = "Highly recommend to anyone considering it.", Label = true },
        new SentimentData { Text = "Wonderful product, exceeded my hopes!", Label = true },
        new SentimentData { Text = "So much better than I expected.", Label = true },
        new SentimentData { Text = "Quality is outstanding!", Label = true },
        new SentimentData { Text = "Completely satisfied, five stars!", Label = true },
        new SentimentData { Text = "A great find, love it!", Label = true },
        new SentimentData { Text = "Highly recommend this to everyone.", Label = true },
        new SentimentData { Text = "I'm extremely satisfied with this purchase.", Label = true },
        new SentimentData { Text = "This exceeded my expectations.", Label = true },
        new SentimentData { Text = "Best thing I've ever owned.", Label = true },
        new SentimentData { Text = "This is exactly what I needed!", Label = true },
        new SentimentData { Text = "I absolutely love this product!", Label = true },
        new SentimentData { Text = "I would definitely buy this again.", Label = true },
        new SentimentData { Text = "This is the best purchase I've made.", Label = true },
        new SentimentData { Text = "Highly satisfied with the results!", Label = true },
        new SentimentData { Text = "Amazing product, works perfectly!", Label = true },
        new SentimentData { Text = "This item exceeded my expectations!", Label = true },
        new SentimentData { Text = "One of the best things I've bought recently.", Label = true },
        new SentimentData { Text = "The quality is outstanding.", Label = true },
        new SentimentData { Text = "Works exactly as advertised.", Label = true },
        new SentimentData { Text = "Very useful and handy!", Label = true },
        new SentimentData { Text = "Exceeded my expectations in every way.", Label = true },
        new SentimentData { Text = "Incredible product! Highly recommended.", Label = true },
        new SentimentData { Text = "Such a wonderful experience using this.", Label = true },
        new SentimentData { Text = "Fantastic performance, exceeded my expectations.", Label = true },
        new SentimentData { Text = "Extremely satisfied with my purchase!", Label = true },
        new SentimentData { Text = "Best product I've used in a long time.", Label = true },
        new SentimentData { Text = "Perfect, just what I was looking for.", Label = true },
        new SentimentData { Text = "Five stars! Highly recommend.", Label = true },
        new SentimentData { Text = "Exceeded all my expectations.", Label = true },
        new SentimentData { Text = "Simply the best product out there.", Label = true },
        new SentimentData { Text = "Impressive build quality and features.", Label = true },
        new SentimentData { Text = "Absolutely love it!", Label = true },
        new SentimentData { Text = "A fantastic product that I enjoy using.", Label = true },
        new SentimentData { Text = "Great for the price!", Label = true },
        new SentimentData { Text = "It works like a charm!", Label = true },
        new SentimentData { Text = "Best decision I’ve made in a while.", Label = true },
        new SentimentData { Text = "Worth every penny!", Label = true },
        new SentimentData { Text = "I am very happy with this purchase.", Label = true },
        new SentimentData { Text = "Excellent quality and value.", Label = true },
        new SentimentData { Text = "Top-notch product, very impressed.", Label = true },
        new SentimentData { Text = "I am thrilled with this product.", Label = true },
        new SentimentData { Text = "Perfect for my needs!", Label = true },
        new SentimentData { Text = "I can't believe how good this is!", Label = true },
        new SentimentData { Text = "Incredible, exceeded my expectations.", Label = true },
        new SentimentData { Text = "A great value for the quality.", Label = true },
        new SentimentData { Text = "Very satisfied with this product.", Label = true },
        new SentimentData { Text = "High quality and reliable.", Label = true },
        new SentimentData { Text = "Absolutely flawless!", Label = true },
        new SentimentData { Text = "This is exactly what I needed.", Label = true },
        new SentimentData { Text = "Best purchase I’ve made in a long time.", Label = true },
        new SentimentData { Text = "I'm very pleased with this item.", Label = true },
        new SentimentData { Text = "This is amazing, highly recommend.", Label = true },
        new SentimentData { Text = "I really enjoy using this.", Label = true },
        new SentimentData { Text = "This product is outstanding.", Label = true },
        new SentimentData { Text = "The best investment I made.", Label = true },
        new SentimentData { Text = "This product is worth every cent.", Label = true },
        new SentimentData { Text = "Very happy with the results.", Label = true },
        new SentimentData { Text = "Excellent, would buy again.", Label = true },
        new SentimentData { Text = "Great experience, highly recommended.", Label = true },
        new SentimentData { Text = "Very well made and durable.", Label = true },new SentimentData { Text = "This product is a game-changer!", Label = true },
        new SentimentData { Text = "Absolutely phenomenal experience!", Label = true },
        new SentimentData { Text = "I would give this ten stars if I could.", Label = true },
        new SentimentData { Text = "Exactly as described, couldn’t be happier.", Label = true },
        new SentimentData { Text = "One of the best purchases I’ve made.", Label = true },
        new SentimentData { Text = "I’m delighted with this product!", Label = true },
        new SentimentData { Text = "The performance is top-notch.", Label = true },
        new SentimentData { Text = "Absolutely worth every penny.", Label = true },
        new SentimentData { Text = "I’m amazed by the quality!", Label = true },
        new SentimentData { Text = "This is everything I hoped it would be.", Label = true },
        new SentimentData { Text = "Superb! Would buy again without hesitation.", Label = true },
        new SentimentData { Text = "Couldn’t be more pleased with this purchase.", Label = true },
        new SentimentData { Text = "An absolute must-have product!", Label = true },
        new SentimentData { Text = "Outstanding craftsmanship and quality.", Label = true },
        new SentimentData { Text = "So glad I bought this, fantastic choice.", Label = true },
        new SentimentData { Text = "Beyond my expectations in every way.", Label = true },
        new SentimentData { Text = "Perfect for what I was looking for.", Label = true },
        new SentimentData { Text = "I highly recommend this to everyone.", Label = true },
        new SentimentData { Text = "I’m thrilled with how well this works.", Label = true },
        new SentimentData { Text = "Simply put, this product is amazing.", Label = true },
        new SentimentData { Text = "The quality and value are unbeatable.", Label = true },
        new SentimentData { Text = "Five-star experience with this product.", Label = true },
        new SentimentData { Text = "This item has truly exceeded my hopes.", Label = true },
        new SentimentData { Text = "An exceptional product I’ll use daily.", Label = true },
        new SentimentData { Text = "This has made my life so much easier.", Label = true },
        new SentimentData { Text = "Highly functional and beautifully designed.", Label = true },
        new SentimentData { Text = "A purchase I’ll never regret.", Label = true },
        new SentimentData { Text = "Perfect in every single way.", Label = true },
        new SentimentData { Text = "This has become one of my favorites.", Label = true },
        new SentimentData { Text = "Unquestionably the best product I own.", Label = true },
        new SentimentData { Text = "Can’t imagine my life without this now.", Label = true },
        new SentimentData { Text = "Highly effective and easy to use.", Label = true },
        new SentimentData { Text = "Such a useful and reliable product.", Label = true },
        new SentimentData { Text = "So happy with the performance of this.", Label = true },
        new SentimentData { Text = "Remarkably high quality and value.", Label = true },
        new SentimentData { Text = "Feels premium, works brilliantly.", Label = true },
        new SentimentData { Text = "It’s perfect, everything I wanted!", Label = true },
        new SentimentData { Text = "The best in its category, hands down.", Label = true },
        new SentimentData { Text = "I’m very impressed by its efficiency.", Label = true },
        new SentimentData { Text = "The attention to detail is superb.", Label = true },
        new SentimentData { Text = "This was worth every dollar spent.", Label = true },
        new SentimentData { Text = "Fantastic quality and ease of use.", Label = true },
        new SentimentData { Text = "I’m blown away by how well it works.", Label = true },
        new SentimentData { Text = "It’s everything I dreamed it would be.", Label = true },
        new SentimentData { Text = "Exactly what I was hoping for!", Label = true },
        new SentimentData { Text = "Works like a dream, couldn’t be happier.", Label = true },
        new SentimentData { Text = "Simply incredible value and quality.", Label = true },
        new SentimentData { Text = "This has brought me so much joy.", Label = true },
        new SentimentData { Text = "Couldn’t ask for a better product.", Label = true },
        new SentimentData { Text = "I’m beyond satisfied with this.", Label = true },
        new SentimentData { Text = "A flawless experience from start to finish.", Label = true },

        // Negative samples
        new SentimentData { Text = "I did not like this product at all.", Label = false },
        new SentimentData { Text = "Worst product I have ever bought.", Label = false },
        new SentimentData { Text = "Terrible, would not buy again.", Label = false },
        new SentimentData { Text = "Would not recommend to anyone.", Label = false },
        new SentimentData { Text = "Very disappointing experience.", Label = false },
        new SentimentData { Text = "Awful, not worth the money.", Label = false },
        new SentimentData { Text = "Worst purchase ever", Label = false },
        new SentimentData { Text = "This was a complete waste of money.", Label = false },
        new SentimentData { Text = "I regret buying this.", Label = false },
        new SentimentData { Text = "I don’t see the hype around this.", Label = false },
        new SentimentData { Text = "Not worth the price.", Label = false },
        new SentimentData { Text = "I'm not impressed with this.", Label = false },
        new SentimentData { Text = "This product is a total disaster.", Label = false },
        new SentimentData { Text = "Completely useless and a waste of money.", Label = false },
        new SentimentData { Text = "This item is so poorly made.", Label = false },
        new SentimentData { Text = "I would give this zero stars if I could.", Label = false },
        new SentimentData { Text = "Such a disappointment, don't buy it.", Label = false },
        new SentimentData { Text = "It broke after just one use, terrible quality.", Label = false },
        new SentimentData { Text = "Completely not what I expected, very let down.", Label = false },
        new SentimentData { Text = "This product feels cheap and flimsy.", Label = false },
        new SentimentData { Text = "This was a complete waste.", Label = false },
        new SentimentData { Text = "Worst experience ever.", Label = false },
        new SentimentData { Text = "Cheap and unreliable.", Label = false },
        new SentimentData { Text = "Not worth a single penny.", Label = false },
        new SentimentData { Text = "A total failure.", Label = false },
        new SentimentData { Text = "Very disappointing.", Label = false },
        new SentimentData { Text = "Complete letdown.", Label = false },
        new SentimentData { Text = "Do not waste your money on this.", Label = false },
        new SentimentData { Text = "It's awful.", Label = false },
        new SentimentData { Text = "Extremely disappointing.", Label = false },
        new SentimentData { Text = "Absolutely horrible experience.", Label = false },
        new SentimentData { Text = "It fell apart quickly.", Label = false },
        new SentimentData { Text = "Terrible customer service and product.", Label = false },
        new SentimentData { Text = "Did not meet my expectations.", Label = false },
        new SentimentData { Text = "Not happy with this at all.", Label = false },
        new SentimentData { Text = "Very poor craftsmanship.", Label = false },
        new SentimentData { Text = "Terrible choice, regret buying it.", Label = false },
        new SentimentData { Text = "Waste of resources.", Label = false },
        new SentimentData { Text = "Totally unreliable.", Label = false },
        new SentimentData { Text = "Subpar experience overall.", Label = false },
        new SentimentData { Text = "Would not buy this again.", Label = false },
        new SentimentData { Text = "An utter disappointment.", Label = false },
        new SentimentData { Text = "Regret spending money on this.", Label = false },
        new SentimentData { Text = "Doesn't live up to the hype.", Label = false },
        new SentimentData { Text = "Completely useless.", Label = false },
        new SentimentData { Text = "This is not worth the hassle.", Label = false },
        new SentimentData { Text = "Poor performance and build quality.", Label = false },
        new SentimentData { Text = "I don't see why anyone would buy this.", Label = false },
        new SentimentData { Text = "It failed to work as promised.", Label = false },
        new SentimentData { Text = "Such a waste, would not buy again.", Label = false },
        new SentimentData { Text = "Highly disappointing purchase.", Label = false },
        new SentimentData { Text = "Absolutely not worth it.", Label = false },
        new SentimentData { Text = "Quality is below average.", Label = false },
        new SentimentData { Text = "Disappointed in every aspect.", Label = false },
        new SentimentData { Text = "It’s defective and useless.", Label = false },
        new SentimentData { Text = "The worst decision I've made.", Label = false },
        new SentimentData { Text = "I regret ever buying this.", Label = false },
        new SentimentData { Text = "This is a terrible item.", Label = false },
        new SentimentData { Text = "Waste of money and time.", Label = false },
        new SentimentData { Text = "I had a bad experience with this.", Label = false },
        new SentimentData { Text = "Regretful purchase.", Label = false },
        new SentimentData { Text = "Extremely poor quality.", Label = false },
        new SentimentData { Text = "Very cheaply made.", Label = false },
        new SentimentData { Text = "One of my worst buys.", Label = false },
        new SentimentData { Text = "Terrible experience, would not recommend.", Label = false },
        new SentimentData { Text = "This item did not work as expected.", Label = false },
        new SentimentData { Text = "Terrible quality, broke after a day.", Label = false },
        new SentimentData { Text = "The product was okay, but nothing special.", Label = false },
        new SentimentData { Text = "Awful, I would never buy this again.", Label = false },
        new SentimentData { Text = "Not worth the money at all.", Label = false },
        new SentimentData { Text = "This was a huge disappointment.", Label = false },
        new SentimentData { Text = "Would not recommend this to anyone.", Label = false },
        new SentimentData { Text = "Regret buying this, total waste.", Label = false },
        new SentimentData { Text = "Poorly made, feels cheap.", Label = false },
        new SentimentData { Text = "Not what I expected, disappointed.", Label = false },
        new SentimentData { Text = "Unreliable and disappointing.", Label = false },
        new SentimentData { Text = "Did not meet my expectations.", Label = false },
        new SentimentData { Text = "Would not purchase again.", Label = false },
        new SentimentData { Text = "This is unacceptable, waste of money.", Label = false },
        new SentimentData { Text = "Feels very low quality.", Label = false },
        new SentimentData { Text = "Does not work as advertised.", Label = false },
        new SentimentData { Text = "This is a joke, it broke immediately.", Label = false },
        new SentimentData { Text = "Not good, poor experience.", Label = false },
        new SentimentData { Text = "Worst purchase I've made in a while.", Label = false },
        new SentimentData { Text = "Terrible design, very disappointed.", Label = false },
        new SentimentData { Text = "Very dissatisfied, returning it.", Label = false },
        new SentimentData { Text = "Waste of money, do not buy.", Label = false },
        new SentimentData { Text = "Very poor performance, regret it.", Label = false },
        new SentimentData { Text = "Absolutely terrible experience.", Label = false },
        new SentimentData { Text = "Not worth the high price.", Label = false },
        new SentimentData { Text = "I expected much more.", Label = false },
        new SentimentData { Text = "This is complete garbage.", Label = false },
        new SentimentData { Text = "Disappointed in the quality.", Label = false },
        new SentimentData { Text = "Did not work as promised.", Label = false },
        new SentimentData { Text = "This is cheaply made.", Label = false },
        new SentimentData { Text = "Fell apart almost immediately.", Label = false },
        new SentimentData { Text = "The worst, absolute waste.", Label = false },
        new SentimentData { Text = "Very disappointed, would not buy.", Label = false },
        new SentimentData { Text = "Not worth buying.", Label = false },
        new SentimentData { Text = "Awful quality, save your money.", Label = false },
        new SentimentData { Text = "Broke right after the return period.", Label = false },
        new SentimentData { Text = "Terrible, doesn't work at all.", Label = false },
        new SentimentData { Text = "This was a complete letdown.", Label = false },
        new SentimentData { Text = "Would not buy again, poor quality.", Label = false },new SentimentData { Text = "I regret wasting my money on this.", Label = false },
        new SentimentData { Text = "Horrible quality, never buying this again.", Label = false },
        new SentimentData { Text = "It broke within days, terrible durability.", Label = false },
        new SentimentData { Text = "Not even close to what I expected.", Label = false },
        new SentimentData { Text = "Such a disappointing experience overall.", Label = false },
        new SentimentData { Text = "Feels cheaply made and low quality.", Label = false },
        new SentimentData { Text = "Terrible, waste of money.", Label = false },
        new SentimentData { Text = "Extremely dissatisfied with the purchase.", Label = false },
        new SentimentData { Text = "I was let down by the quality of this item.", Label = false },
        new SentimentData { Text = "Did not work as described, complete failure.", Label = false },
        new SentimentData { Text = "I am very unhappy with this product.", Label = false },
        new SentimentData { Text = "Looks nothing like advertised, very disappointed.", Label = false },
        new SentimentData { Text = "Stopped working after one use, horrible experience.", Label = false },
        new SentimentData { Text = "One of the worst products I've ever used.", Label = false },
        new SentimentData { Text = "Disgustingly poor quality.", Label = false },
        new SentimentData { Text = "Absolutely no value for money.", Label = false },
        new SentimentData { Text = "Cheaply made, very fragile.", Label = false },
        new SentimentData { Text = "I wouldn’t recommend this to anyone.", Label = false },
        new SentimentData { Text = "Worst product I've purchased in years.", Label = false },
        new SentimentData { Text = "The quality is substandard.", Label = false },
        new SentimentData { Text = "Completely fell apart within a week.", Label = false },
        new SentimentData { Text = "Overpriced and useless.", Label = false },
        new SentimentData { Text = "Not as advertised, felt cheated.", Label = false },
        new SentimentData { Text = "This product is a nightmare.", Label = false },
        new SentimentData { Text = "Barely functions, complete disaster.", Label = false },
        new SentimentData { Text = "Such a waste, I regret buying this.", Label = false },
        new SentimentData { Text = "Didn’t meet any of my expectations.", Label = false },
        new SentimentData { Text = "Feels like it was made from cheap materials.", Label = false },
        new SentimentData { Text = "Broke down within hours of use.", Label = false },
        new SentimentData { Text = "I am extremely disappointed with this.", Label = false },
        new SentimentData { Text = "Very low quality, wouldn’t buy again.", Label = false },
        new SentimentData { Text = "Useless item, doesn’t work as claimed.", Label = false },
        new SentimentData { Text = "Shoddy construction, very poorly made.", Label = false },
        new SentimentData { Text = "One of the worst purchases I've ever made.", Label = false },
        new SentimentData { Text = "Cheap and low-grade product.", Label = false },
        new SentimentData { Text = "I feel ripped off by this purchase.", Label = false },
        new SentimentData { Text = "Failed to deliver on its promises.", Label = false },
        new SentimentData { Text = "Terrible design, uncomfortable to use.", Label = false },
        new SentimentData { Text = "I wouldn’t waste my money on this.", Label = false },
        new SentimentData { Text = "Extremely poor value, total letdown.", Label = false },
        new SentimentData { Text = "Not durable at all, broke immediately.", Label = false },
        new SentimentData { Text = "Completely disappointed with the quality.", Label = false },
        new SentimentData { Text = "Doesn’t function properly, very frustrating.", Label = false },
        new SentimentData { Text = "The worst product I’ve come across.", Label = false },
        new SentimentData { Text = "Cheap materials, feels like a toy.", Label = false },
        new SentimentData { Text = "Extremely flimsy and poorly made.", Label = false },
        new SentimentData { Text = "Not worth the purchase, poor quality.", Label = false },
        new SentimentData { Text = "The product fell short of my expectations.", Label = false },
        new SentimentData { Text = "I would never recommend this to anyone.", Label = false },
        new SentimentData { Text = "It’s poorly designed and ineffective.", Label = false },
        new SentimentData { Text = "Broke after a single use, very upset.", Label = false },
        new SentimentData { Text = "Complete waste, no redeeming qualities.", Label = false },
        new SentimentData { Text = "The product quality is very poor.", Label = false },
        new SentimentData { Text = "Feels like a cheap knock-off.", Label = false },
        new SentimentData { Text = "Not functional at all, very frustrating.", Label = false },
        new SentimentData { Text = "Regretted this purchase the moment I used it.", Label = false },
        new SentimentData { Text = "Not worth a single cent.", Label = false },
        new SentimentData { Text = "Such poor quality, it’s unusable.", Label = false },
        new SentimentData { Text = "Stopped working almost immediately.", Label = false },
        new SentimentData { Text = "Definitely not worth the cost.", Label = false },
        new SentimentData { Text = "Feels extremely cheap, very flimsy.", Label = false }
    };

    var positives = trainingData.Count(data => data.Label == true);
    var negatives = trainingData.Count(data => data.Label == false);

    Console.WriteLine($"Positive samples: {positives}, Negative samples: {negatives}");

    return mlContext.Data.LoadFromEnumerable(trainingData);
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
}
