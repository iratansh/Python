//
//  ContentView.swift
//  Stock Predictor
//
//  Created by Ishaan Ratanshi on 2024-03-24.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        ZStack{
            LinearGradient(gradient: Gradient(colors: [.black, .white]), startPoint: .topLeading, endPoint: .bottomTrailing)
                .edgesIgnoringSafeArea(.all)
            VStack {
                Text("Stock Predictor")
                    .font(.system(size: 32, weight: .medium, design: .default))
                    .foregroundColor(.white)
                    .frame(width: 200, height: 200, alignment: .center)
                Spacer()
            }
        }
    }
}

#Preview {
    ContentView()
}
