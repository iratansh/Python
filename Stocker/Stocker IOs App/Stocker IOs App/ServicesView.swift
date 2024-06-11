// ServiceView component for Navbar

import SwiftUI

struct ServicesView: View {
    var body: some View {
        NavigationView {
            GeometryReader { geometry in
                ScrollView {
                    VStack { 
                        Text("Services")
                            .font(.system(size: 32, weight: .medium))
                            .foregroundColor(.black)
                            .padding(.top, 20)
                        Text("Stock Forecasting using Stocker is limited to AAPL, GOOGL, MSFT, SPOT, TSLA, VTI")
                            .font(.system(size: 16))
                            .foregroundColor(.black)
                            .padding()
                            .frame(width: geometry.size.width * 0.8) 
                    }
                    .frame(maxWidth: .infinity, alignment: .center) 
                }
            }
        }
    }
}
