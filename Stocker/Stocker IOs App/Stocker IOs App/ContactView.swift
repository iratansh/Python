// For ContactView component accessed through Navbar

import SwiftUI

struct ContactView: View {
    var body: some View {
        VStack {
            Spacer()
            
            Text("Contact")
                .font(.system(size: 32, weight: .medium))
                .foregroundColor(.black)
                .padding(.bottom, 20)
            
            Text("Email: iratansh@ualberta.ca")
                .font(.system(size: 20))
                .foregroundColor(.black)
            
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.white)
        .edgesIgnoringSafeArea(.all)
    }
}

struct ContactView_Previews: PreviewProvider {
    static var previews: some View {
        ContactView()
    }
}
