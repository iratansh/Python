// Main SwiftUI app component

import SwiftUI
import Combine

struct ContentView: View {
    @State private var selectedStock: String = "Select a stock"
    @State private var showHome: Bool = false
    @State private var showAbout: Bool = false
    @State private var showServices: Bool = false
    @State private var showContact: Bool = false
    @State private var predictionResult: [String: Any]?
    @State private var isPredicting: Bool = false
    @State private var isLoading: Bool = true

    var body: some View {
        NavigationView {
            ZStack {
                VStack {
                    if !isLoading {
                        if showHome {
                            HomeView(selectedStock: $selectedStock, predictionResult: $predictionResult, isPredicting: $isPredicting, isLoading: $isLoading)
                        } else if showAbout {
                            AboutView()
                        } else if showServices {
                            ServicesView()
                        } else if showContact {
                            ContactView()
                        } else {
                            // Display HomeView by default
                            HomeView(selectedStock: $selectedStock, predictionResult: $predictionResult, isPredicting: $isPredicting, isLoading: $isLoading)
                        }
                        
                        Spacer()

                        // Navbar
                        HStack {
                            Spacer()
                            Button(action: {
                                showHome = true
                                showAbout = false
                                showServices = false
                                showContact = false
                            }) {
                                VStack {
                                    Image(systemName: "house")
                                    Text("Home")
                                        .font(.system(size: 12))
                                }
                            }
                            .foregroundColor(.black)
                            Spacer()
                            Button(action: {
                                showHome = false
                                showAbout = true
                                showServices = false
                                showContact = false
                            }) {
                                VStack {
                                    Image(systemName: "questionmark.circle")
                                    Text("About")
                                        .font(.system(size: 12))
                                }
                            }
                            .foregroundColor(.black)
                            Spacer()
                            Button(action: {
                                showHome = false
                                showAbout = false
                                showServices = true
                                showContact = false
                            }) {
                                VStack {
                                    Image(systemName: "doc.text")
                                    Text("Services")
                                        .font(.system(size: 12))
                                }
                            }
                            .foregroundColor(.black)
                            Spacer()
                            Button(action: {
                                showHome = false
                                showAbout = false
                                showServices = false
                                showContact = true
                            }) {
                                VStack {
                                    Image(systemName: "envelope")
                                    Text("Contact")
                                        .font(.system(size: 12))
                                }
                            }
                            .foregroundColor(.black)
                            Spacer()
                        }
                        .padding(.top, 10)
                        .padding(.bottom, 20)
                        .background(Color.gray.opacity(0.1))
                        .overlay(Rectangle().frame(height: 1).foregroundColor(.black), alignment: .top)
                        .edgesIgnoringSafeArea(.bottom)
                    }
                }
                
                // Loading screen
                if isLoading {
                    ZStack {
                        Color.white
                            .edgesIgnoringSafeArea(.all)
                        LoadingView()
                    }
                    .onAppear {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                            withAnimation {
                                isLoading = false
                            }
                        }
                    }
                }
                
                // Prediction progress indicator
                if isPredicting {
                    ZStack {
                        Color.black.opacity(0.5)
                            .edgesIgnoringSafeArea(.all)
                        ProgressView("Predicting...")
                            .padding(20)
                            .background(Color.white)
                            .cornerRadius(10)
                    }
                }
            }
            .onReceive(Just(predictionResult)) { prediction in
                if prediction != nil {
                    // Clear everything except the Navbar when the predictions are fetched
                    showHome = false
                    showAbout = false
                    showServices = false
                    showContact = false
                }
            }
        }
    }
}


#Preview {
    ContentView()
}


