import "@/styles/globals.css";
import './Home.css';
import type { AppProps } from "next/app";
import { NextUIProvider } from '@nextui-org/react';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <div className='background'>
    <NextUIProvider>
      <Component {...pageProps} />
    </NextUIProvider>
    </div>
  );
}
