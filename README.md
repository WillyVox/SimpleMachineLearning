
### Simple Machine Learning
    Build a simple machine learning project from scratch using ML.NET

    1. Step 1: Create the Project:
        Run the following .NET CLI command to create a new console application:
        $ dotnet new console -n SimpleMachineLearning

    2. Step 2: Add Required ML.NET NuGet Packages
        1. Add Microsoft.ML:
        This is the core ML.NET library.
        $ dotnet add package Microsoft.ML

        2. Add a Specific Trainer Package (e.g., Microsoft.ML.FastTree):
        Since the example uses FastTreeRegressionTrainer, you need its package.
        $ dotnet add package Microsoft.ML.FastTree
        
        3. Add Logging Packages (for console output):
        These are needed for the ILogger and LoggerFactory used in the Program.cs to show detailed logs.
        $ dotnet add package Microsoft.Extensions.Logging
        $ dotnet add package Microsoft.Extensions.Logging.Console

    3. Step 3: Replace Program.cs Content

        1. Open the Project:
        You can open the SimpleMLProject folder in Visual Studio Code (code .) or Visual Studio.

        2. Replace Program.cs:
        Open the Program.cs file in your new SimpleMLProject and replace its entire content with the C# code provided above in the <immersive id="ml-project-code-new" ...> block.

    4. Step 4: Run the Application

        1. Build the Project:
        In your terminal, within the SimpleMLProject directory, run:
        $ dotnet build
        This will compile your code and download any necessary dependencies.

        2. Run the Application:
        After a successful build, run the application:
        $ dotnet run

    5. Conclusion
        You should see output similar to this (the exact predicted prices might vary slightly depending on the ML.NET version and specific algorithm implementations, but will be close to the linear trend):

        info: SimpleMLProject.Program[0]
            Starting Simple ML.NET Project...
        info: SimpleMLProject.Program[0]
            Loading sample training data...
        info: SimpleMLProject.Program[0]
            Building and training the ML.NET model...
        info: SimpleMLProject.Program[0]
            ML.NET model training completed successfully.
        info: SimpleMLProject.Program[0]
            Predicted price for a 700 sqft house: $115,000.00
        info: SimpleMLProject.Program[0]
            Predicted price for a 1300 sqft house: $195,000.00
        info: SimpleMLProject.Program[0]
            Predicted price for a 2500 sqft house: $335,000.00
        
        Simple ML.NET Project finished.
        
        Press any key to exit...
        This console application now contains a fully functional, simple machine learning project using ML.NET for linear regression, built from scratch!