package com.example.demo.controller;

import java.util.ArrayList;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

import com.example.demo.mapper.DetailMapper;
import com.example.demo.mapper.EventListMapper;
import com.example.demo.model.DetailList;
import com.example.demo.model.EventList;

import jakarta.servlet.http.HttpSession;

@Controller
public class HomeController {

    @Autowired
    DetailMapper detailMapper;

    @Autowired
    EventListMapper eventListMapper;

    @GetMapping("/")
    public String mainPage (HttpSession session, Model model){
        session.invalidate();
        return "main";
    }

    @GetMapping("userPage")
    public String userPage (HttpSession session, Model model){

        ArrayList<EventList> eventList = eventListMapper.selectAll();
        System.out.println(eventList);
        model.addAttribute("eventList", eventList);

        return "userPage";
    }

    @GetMapping("detailPage")
    public String detailPage (HttpSession session, Model model, @RequestParam String event_name){

        ArrayList<DetailList> detailList = detailMapper.select(event_name);
        model.addAttribute("detailList", detailList);
        model.addAttribute("event_name", event_name);

        return "detailPage";
    }

    
}
