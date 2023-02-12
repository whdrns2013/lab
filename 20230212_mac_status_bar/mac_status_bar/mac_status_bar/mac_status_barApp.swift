//
//  mac_status_barApp.swift
//  mac_status_bar
//
//  Created by Jongya on 2023/02/12.
//

import SwiftUI
import mach
#include <mach/host_info.h>

@main
struct mac_status_barApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}



let HOST_CPU_LOAD_INFO_COUNT = UInt32(MemoryLayout<host_cpu_load_info>.size / MemoryLayout<integer_t>.size)
var cpuLoadInfo = host_cpu_load_info()
var size = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info>.stride / MemoryLayout<integer_t>.stride)
let result = withUnsafeMutablePointer(to: &cpuLoadInfo) {
    host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, host_info_t($0), &size)
}

class ViewController: NSViewController {
    @IBOutlet weak var cpuLabel: NSTextField!
    @IBOutlet weak var memoryLabel: NSTextField!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        updateStatus()
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { (timer) in
            self.updateStatus()
        }
    }
    
    func updateStatus() {
        let host = host_info()
        let cpuUsage = cpuUsagePercentage(host: host)
        let memoryUsage = memoryUsagePercentage(host: host)
        DispatchQueue.main.async {
            self.cpuLabel.stringValue = "CPU: \(cpuUsage)%"
            self.memoryLabel.stringValue = "Memory: \(memoryUsage)%"
        }
    }


    func host_info() -> host_cpu_load_info {
        var size = mach_msg_type_number_t(HOST_CPU_LOAD_INFO_COUNT)
        let host = mach_host_self()
        var hostInfo = host_cpu_load_info()
        let result = host_statistics(host, HOST_CPU_LOAD_INFO,
                                     UnsafeMutablePointer(&hostInfo), &size)
        guard result == KERN_SUCCESS else {
            fatalError("Error \(result)")
        }
        return hostInfo
    }
    
    func cpuUsagePercentage(host: host_cpu_load_info) -> Int {
        let totalTicks = host.cpu_ticks.0 + host.cpu_ticks.1 + host.cpu_ticks.2
        let idleTicks = host.cpu_ticks.0
        let totalLoad = totalTicks - idleTicks
        let percentage = 100.0 * Double(totalLoad) / Double(totalTicks)
        return Int(round(percentage))
    }
    
    func memoryUsagePercentage(host: host_cpu_load_info) -> Int {
        let totalMemory = NSProcessInfo.processInfo.physicalMemory
        let usedMemory = totalMemory - NSProcessInfo.processInfo.freeMemory
        let percentage = 100.0 * Double(usedMemory) / Double(totalMemory)
        return Int(round(percentage))
    }
}
