import SwiftUI

struct ContentView: View {
    @State private var selectedStock: String = "Select a stock"
    @State private var stocks: [String] = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    
    var body: some View {
        VStack {
            ZStack {
                Color.white
                    .edgesIgnoringSafeArea(.all)
                VStack {
                    Spacer().frame(height: 50) // Add some space at the top
                    
                    Text("Stocker")
                        .font(.system(size: 32, weight: .medium, design: .default))
                        .foregroundColor(.black)
                        .frame(height: 50, alignment: .center)
                    
                    VStack(spacing: 16) {
                        Text("Select a stock:")
                            .font(.system(size: 18, weight: .medium, design: .default))
                            .foregroundColor(.black)
                        
                        Menu {
                            ForEach(stocks, id: \.self) { stock in
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
                    }
                    .padding()
                    
                    Spacer()
                }
            }
            
            // Fixed Navbar at the bottom
            HStack {
                Spacer()
                Button(action: {
                    // Action for Home
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
                    // Action for Question
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
                    // Action for Paper
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
                    // Action for Envelope
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
            .edgesIgnoringSafeArea(.bottom)
        }
    }
}

#Preview {
    ContentView()
}
