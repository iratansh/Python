// For About Component accessed through Navbar

import SwiftUI

struct AboutView: View {
    var body: some View {
        NavigationView {
            GeometryReader { geometry in
                ScrollView {
                    VStack {
                        Text("About")
                            .font(.system(size: 32, weight: .medium))
                            .foregroundColor(.black)
                            .padding(.top, 20)
                        Text("Stocker utilizes advanced machine learning methods like XGBoost and Bayesian Neural Networks to forecast stock prices over the upcoming seven days. It utilizes the latest stock data from Yahoo Finance, to deliver the most accurate predictions. The application boasts user-friendly interfaces developed with Next.js and Swift, ensuring seamless access to dynamic market predictions. Technologies employed include Python, Swift, TypeScript, TailwindCSS, Next.js, Node.js, Pyro, Flask, XGBoost, Optuna, pandas, and Matplotlib.")
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

