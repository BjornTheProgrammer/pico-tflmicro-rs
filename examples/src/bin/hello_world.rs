#![no_std]
#![no_main]

use embassy_executor::Spawner;
use embassy_rp::peripherals::USB;

use embassy_rp::bind_interrupts;
use embassy_rp::gpio;
use embassy_rp::rom_data::reset_to_usb_boot;
use embassy_rp::usb::{Driver, InterruptHandler as UsbInterruptHandler};
use embassy_time::Timer;
use gpio::{Level, Output};
use log::*;
use {defmt_rtt as _, panic_probe as _};
use pico_tflmicro_sys::*;

bind_interrupts!(struct Irqs {
    USBCTRL_IRQ => UsbInterruptHandler<USB>;
});

use usb::*;
// mod serial;

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());
    let driver = Driver::new(p.USB, Irqs);
    spawner.spawn(usb_task(driver)).unwrap();

    let mut led = Output::new(p.PIN_25, Level::Low);
    let mut bootsel = p.BOOTSEL;
    led.set_high();

    Timer::after_secs(2).await;

    info!("started");

    let model = include_bytes!("../../models/hello_world.tflite");
    let buf = &model[..];
    let len = buf.len();
    let buf = buf.as_ptr();

    unsafe {
        let model = getModel(*buf as *const cty::c_void, len);
        let resolver = createEmptyResolver();
        let status = addResolver(resolver, AddSqrt);
        let interpreter = getInterpreter(model, resolver, 2000);

        let input = getTensorInput(interpreter, 0);
    };

    // let mut level = 0;
    loop {
        if bootsel.is_pressed() {
            reset_to_usb_boot(0, 0);
        }
        Timer::after_millis(10).await;
    }
}
