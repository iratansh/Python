// HomeView Component - main display for app

import SwiftUI
import Combine

struct HomeView: View {
    @Binding var selectedStock: String
    @Binding var predictionResult: [String: Any]?
    @Binding var isPredicting: Bool
    @Binding var isLoading: Bool

    var body: some View {
        NavigationView {
            VStack {
                if let prediction = predictionResult?["prediction"] as? [Double], let graphPath = predictionResult?["graph_path"] as? String {
                    HStack {
                        NavigationLink(destination: HomeView(
                            selectedStock: $selectedStock,
                            predictionResult: .constant(nil),
                            isPredicting: $isPredicting,
                            isLoading: $isLoading
                        )) {
                            HStack {
                                Image(systemName: "chevron.left")
                                Text("Back")
                            }
                            .foregroundColor(.blue)
                        }
                        Spacer()
                    }
                    .padding()

                    Text("Predictions for \(selectedStock)")
                        .font(.title)
                        .padding()

                    PredictionResultView(prediction: prediction)

                    if let url = URL(string: "http://localhost:5001\(graphPath)") {
                        AsyncImage(url: url) { image in
                            image
                                .resizable()
                                .scaledToFit()
                                .frame(maxWidth: 375, maxHeight: 350)
                                .padding(.top, 20)
                        } placeholder: {
                            ProgressView()
                                .frame(width: 375, height: 350)
                        }
                    }
                
                    
                } else {
                    VStack {
                        Spacer().frame(height: 50) 

                        Text("Stocker")
                            .font(.system(size: 32, weight: .medium, design: .default))
                            .foregroundColor(.black)
                            .frame(height: 50, alignment: .center)

                        VStack(spacing: 16) {
                            Text("Select a stock:")
                                .font(.system(size: 18, weight: .medium, design: .default))
                                .foregroundColor(.black)

                            Menu {
                                ForEach(["AAPL", "GOOGL", "MSFT", "SPOT", "TSLA", "VTI"], id: \.self) { stock in
                                    Button(action: {
                                        selectedStock = stock
                                    }) {
                                        Text(stock)
                                    }
                                }
                            } label: {
                                HStack {
                                    Text(selectedStock)
                                        .font(.system(size: 16))
                                        .foregroundColor(.black)
                                    Spacer()
                                    Image(systemName: "chevron.down")
                                        .foregroundColor(.black)
                                }
                                .padding()
                                .background(Color.gray.opacity(0.2))
                                .cornerRadius(8)
                                .frame(width: 200, height: 50)
                            }

                            Button(action: {
                                // Start the prediction
                                isPredicting = true
                                fetchPrediction(for: selectedStock, isLoading: $isLoading) { (result, error) in
                                    DispatchQueue.main.async {
                                        isPredicting = false
                                        if let result = result {
                                            predictionResult = result
                                        } else if let error = error {
                                            print("Error fetching prediction: \(error)")
                                        }
                                    }
                                }
                            }) {
                                Text("Predict")
                                    .font(.system(size: 16, weight: .medium, design: .default))
                                    .foregroundColor(.white)
                                    .padding()
                                    .frame(width: 200, height: 50)
                                    .background(Color.blue)
                                    .cornerRadius(8)
                            }
                        }
                        .padding()

                        Spacer()
                    }
                }
            }
        }
    }
}

struct PredictionResultView: View {
    var prediction: [Double]

    var body: some View {
        VStack(spacing: 16) {
            ForEach(0..<prediction.count, id: \.self) { index in
                Text("Day \(index + 1): \(prediction[index])")
                    .font(.system(size: 16))
                    .foregroundColor(.black)
            }
        }
        .padding()
        .background(Color.gray.opacity(0.2))
        .cornerRadius(8)
        .frame(width: 300)
    }
}

func fetchPrediction(for stock: String, isLoading: Binding<Bool>, completion: @escaping ([String: Any]?, Error?) -> Void) {
    isLoading.wrappedValue = true

    let urlString = "http://localhost:5001/predict?stock=\(stock)"
    guard let url = URL(string: urlString) else {
        print("Invalid URL: \(urlString)")
        completion(nil, NSError(domain: "InvalidURL", code: 0, userInfo: nil))
        isLoading.wrappedValue = false
        return
    }

    print("Fetching URL: \(urlString)")

    let config = URLSessionConfiguration.default
    config.timeoutIntervalForRequest = 3600 // make the request timeout longer
    let session = URLSession(configuration: config)

    let task = session.dataTask(with: url) { (data, response, error) in
        DispatchQueue.main.async {
            isLoading.wrappedValue = false
        }

        if let error = error {
            print("Error: \(error.localizedDescription)")
            completion(nil, error)
            return
        }

        guard let data = data else {
            print("No data received")
            completion(nil, NSError(domain: "NoData", code: 0, userInfo: nil))
            return
        }

        do {
            if let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
                print("Received JSON: \(json)")
                completion(json, nil)
            } else {
                print("Invalid JSON received")
                completion(nil, NSError(domain: "InvalidResponse", code: 0, userInfo: nil))
            }
        } catch {
            print("JSON parsing error: \(error)")
            completion(nil, error)
        }
    }

    task.resume()
}


#Preview {
    HomeView(selectedStock: .constant("Select a stock"))
}
