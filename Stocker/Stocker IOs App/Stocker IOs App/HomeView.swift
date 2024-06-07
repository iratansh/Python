import SwiftUI

struct HomeView: View {
    @Binding var selectedStock: String
    @State private var stocks: [String] = ["AAPL", "GOOGL", "MSFT", "SPOT", "TSLA", "VTI"]
    @State private var isMenuOpen: Bool = false

    var body: some View {
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
                    ForEach(stocks, id: \.self) { stock in
                        Button(action: {
                            selectedStock = stock
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                                isMenuOpen.toggle()
                            }
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
                            .rotationEffect(.degrees(isMenuOpen ? 180 : 0))
                            .animation(.linear(duration: 0.2), value: isMenuOpen)
                    }
                    .padding()
                    .background(Color.gray.opacity(0.2))
                    .cornerRadius(8)
                    .frame(width: 200, height: 50)
                }
                .onTapGesture {
                    withAnimation {
                        isMenuOpen.toggle()
                    }
                }

                Button(action: {
                    print("Predict button clicked for \(selectedStock)")
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

#Preview {
    HomeView(selectedStock: .constant("Select a stock"))
}
