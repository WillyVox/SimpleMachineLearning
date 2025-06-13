// Import necessary ML.NET and logging namespaces
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;
using System;
using System.Linq;
using System.Collections.Generic;

namespace SimpleMachineLearning
{
    // 1. Define your input data model
    public class HousingData
    {
        public float Size { get; set; } // Input feature: Size of the house in square feet

        [ColumnName("Label")]
        public float Price { get; set; } // Output/target variable: Price of the house
    }

    // 2. Define your prediction output model
    public class PricePrediction
    {
        [ColumnName("Score")] // Ensure ML.NET recognizes this as the output column
        public float PredictedPrice { get; set; }
    }

    public class Program
    {
        private static MLContext _mlContext = new MLContext(seed: 0);
        private static ILogger _logger;

        public static void Main(string[] args)
        {
            using var loggerFactory = LoggerFactory.Create(builder =>
            {
                builder.AddConsole();
                builder.SetMinimumLevel(LogLevel.Information);
            });
            _logger = loggerFactory.CreateLogger<Program>();

            _logger.LogInformation("Starting Simple ML.NET Project...");

            var trainingData = LoadSampleData();
            ITransformer trainedModel = TrainModel(trainingData);

            // Evaluate model performance
            EvaluateModel(trainedModel, trainingData);

            // Make predictions
            TestPrediction(trainedModel, 700);
            TestPrediction(trainedModel, 1300);
            TestPrediction(trainedModel, 2500);

            _logger.LogInformation("Simple ML.NET Project finished.");
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        private static IDataView LoadSampleData()
        {
            _logger.LogInformation("Loading sample training data...");

            var data = new[]
            {
                new HousingData { Size = 600, Price = 100000 },
                new HousingData { Size = 800, Price = 120000 },
                new HousingData { Size = 1000, Price = 150000 },
                new HousingData { Size = 1200, Price = 180000 },
                new HousingData { Size = 1500, Price = 220000 },
                new HousingData { Size = 1800, Price = 250000 },
                new HousingData { Size = 2000, Price = 280000 },
                new HousingData { Size = 2200, Price = 300000 },
                new HousingData { Size = 2400, Price = 320000 }
            };

            return _mlContext.Data.LoadFromEnumerable(data);
        }

        private static ITransformer TrainModel(IDataView trainingData)
        {
            _logger.LogInformation("Building and training the ML.NET model...");

            // var pipeline = _mlContext.Transforms.NormalizeMinMax("Size") // Normalize feature values
            //     .Append(_mlContext.Transforms.Concatenate("Features", "Size"))
            //     .Append(_mlContext.Regression.Trainers.SdcaRegression()); // Use SDCA regression
            var pipeline = _mlContext.Transforms.Concatenate("Features", "Size")
    .Append(_mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features"));


            var trainedModel = pipeline.Fit(trainingData);
            _logger.LogInformation("ML.NET model training completed successfully.");
            return trainedModel;
        }

        private static void EvaluateModel(ITransformer model, IDataView trainingData)
        {
            var predictions = model.Transform(trainingData);
            var metrics = _mlContext.Regression.Evaluate(predictions);

            _logger.LogInformation($"Model Evaluation Metrics:");
            _logger.LogInformation($"R-Squared: {metrics.RSquared}");
            _logger.LogInformation($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
        }

        private static void TestPrediction(ITransformer model, float houseSize)
        {
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<HousingData, PricePrediction>(model);
            var input = new HousingData { Size = houseSize };
            var prediction = predictionEngine.Predict(input);

            _logger.LogInformation($"Predicted price for a {houseSize} sqft house: {prediction.PredictedPrice:C}");
        }
    }
}
