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

struct ContentView: View {
    @State private var isLoading: Bool = true
    @State private var selectedStock: String = "Select a stock"
    @State private var showHome: Bool = false
    @State private var showAbout: Bool = false
    @State private var showServices: Bool = false
    @State private var showContact: Bool = false
    
    var body: some View {
        NavigationView {
            ZStack {
                // Main content
                VStack {
                    if !isLoading {
                        if showHome {
                            HomeView(selectedStock: $selectedStock)
                        } else if showAbout {
                            AboutView()
                        } else if showServices {
                            ServicesView()
                        } else if showContact {
                            ContactView()
                        } else {
                            HomeView(selectedStock: $selectedStock)
                        }
                        
                        Spacer()
                        
                        // Navbar with Home, About, Services, and Contact buttons
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
            }
        }
    }
}





#Preview {
    ContentView()
}

