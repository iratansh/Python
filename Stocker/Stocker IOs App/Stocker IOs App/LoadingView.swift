// LoadingView component for adding loading screen at the launch of the app

import SwiftUI

struct LoadingView: View {
    @State private var dotCount: Int = 1
    
    var body: some View {
        VStack {
            Text("Stocker")
                .font(.system(size: 32, weight: .medium, design: .default))
                .foregroundColor(.black)
            
            HStack(spacing: 0) {
                Text("Loading")
                    .font(.system(size: 18, weight: .medium, design: .default))
                    .foregroundColor(.black)
                Text(String(repeating: ".", count: dotCount))
                    .font(.system(size: 18, weight: .medium, design: .default))
                    .foregroundColor(.black)
                    .onAppear {
                        startLoadingAnimation()
                    }
            }
        }
    }
    
    private func startLoadingAnimation() {
        Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { timer in
            dotCount = (dotCount % 3) + 1
        }
    }
}
