// See https://aka.ms/new-console-template for more information
// Console.WriteLine("Hello, World!");

// Replace by new code
// Import necessary ML.NET and logging namespaces
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;
using System;
using System.Linq;
using System.Collections.Generic;

// Define a namespace for your Simle ML
namespace SimpleMachineLearning
{
    // 1. Define your input data model (schema for training data and prediction input)
    // This class represents the structure of your data.
    public class HousingData
    {
        // [LoadColumn(0)] specifies that this property maps to the first column in your data source.
        public float Size { get; set; } // Input feature: Size of the house in square feet

        // [LoadColumn(1)] maps to the second column.
        // [ColumnName("Label")] explicitly marks this property as the 'label' or target variable
        // that the ML model will learn to predict.
        [ColumnName("Label")]
        public float Price { get; set; } // Output/target variable: Price of the house
    }

    // 2. Define your prediction output model
    // This class holds the prediction result after the model processes new input.
    public class PricePrediction
    {
        // [ColumnName("Score")] is the default name ML.NET gives to the output of a regression model.
        public float PredictedPrice { get; set; }
    }

    // 3. Main class to run the ML project
    public class Program
    {
        // Create an MLContext. This is the starting point for all ML.NET operations.
        // The seed parameter ensures reproducibility of results.
        private static MLContext _mlContext = new MLContext(seed: 0);

        // A static logger for console output, similar to how ASP.NET Core logs.
        private static ILogger _logger;

        public static void Main(string[] args)
        {
            // Configure logging for the console application
            using var loggerFactory = LoggerFactory.Create(builder =>
            {
                builder.AddConsole();
                builder.SetMinimumLevel(LogLevel.Information); // Set minimum level for console output
            });
            _logger = loggerFactory.CreateLogger<Program>();

            _logger.LogInformation("Starting Simple ML.NET Project...");

            // Load sample data
            var trainingData = LoadSampleData();

            // Build and train the model
            ITransformer trainedModel = TrainModel(trainingData);

            // Save the model (optional, but good practice for production)
            // SaveModel(_mlContext, trainedModel, "housePriceModel.zip");

            // Make predictions
            TestPrediction(trainedModel, 700); // Predict price for 700 sqft
            TestPrediction(trainedModel, 1300); // Predict price for 1300 sqft
            TestPrediction(trainedModel, 2500); // Predict price for 2500 sqft

            _logger.LogInformation("Simple ML.NET Project finished.");

            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        // Method to load sample training data.
        // In a real-world scenario, this data would come from a database, CSV, or other sources.
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

            // Load the in-memory data into an IDataView, which is the format ML.NET uses.
            return _mlContext.Data.LoadFromEnumerable(data);
        }

        // Method to build and train the ML.NET model
        private static ITransformer TrainModel(IDataView trainingData)
        {
            _logger.LogInformation("Building and training the ML.NET model...");

            // Define the ML.NET data processing and training pipeline:
            // 1. .Transforms.Concatenate("Features", "Size"):
            //    This step takes the 'Size' column and creates a new column called 'Features'.
            //    In ML.NET, input features (variables used for prediction) must be in a column named "Features"
            //    and typically be a vector type (even for a single numerical feature).
            // 2. .Append(_mlContext.Regression.Trainers.FastTree()):
            //    This appends a regression trainer to the pipeline.
            //    FastTree is an ensemble of decision trees, a robust algorithm for regression tasks.
            //    It will learn the relationship between 'Features' (Size) and 'Label' (Price).
            var pipeline = _mlContext.Transforms.Concatenate("Features", "Size")
                .Append(_mlContext.Regression.Trainers.FastTree());

            // Train the model by fitting the pipeline to the training data.
            // This is where the machine learning algorithm learns from your data.
            var trainedModel = pipeline.Fit(trainingData);

            _logger.LogInformation("ML.NET model training completed successfully.");
            return trainedModel;
        }

        // Method to save the trained model to a file
        private static void SaveModel(MLContext mlContext, ITransformer model, string modelPath)
        {
            _logger.LogInformation($"Saving the trained model to '{modelPath}'...");
            mlContext.Model.Save(model, null, modelPath);
            _logger.LogInformation("Model saved.");
        }

        // Method to make a prediction using the trained model
        private static void TestPrediction(ITransformer model, float houseSize)
        {
            // Create a PredictionEngine. This is a convenience API for making single predictions.
            // It is thread-safe, so you can reuse it across multiple predictions.
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<HousingData, PricePrediction>(model);

            // Create an input instance with the house size you want to predict for
            var input = new HousingData { Size = houseSize };

            // Make the prediction
            var prediction = predictionEngine.Predict(input);

            // Log the prediction result. ":C" formats the float as currency based on current culture.
            _logger.LogInformation($"Predicted price for a {houseSize} sqft house: {prediction.PredictedPrice:C}");
        }
    }
}
