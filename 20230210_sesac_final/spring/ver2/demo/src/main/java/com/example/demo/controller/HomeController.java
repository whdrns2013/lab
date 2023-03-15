package com.example.demo.controller;

import java.util.ArrayList;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;

import com.example.demo.mapper.DetailMapper;
import com.example.demo.model.EventList;


import jakarta.servlet.http.HttpSession;

@Controller
public class HomeController {

    @Autowired
    DetailMapper eventListMapper;


    @GetMapping("/")
    public String main (HttpSession session){
        session.invalidate();
        return "main";
    }

    @GetMapping("/userPage")
    public String userPage (Model model, HttpSession session){

        ArrayList<EventList> eventList = eventListMapper.selectAll();
        model.addAttribute("eventList", eventList);
        System.out.println("event list : " + eventList);

        return "userPage";
    }

    @GetMappiing("/detail")
    public String detail (@RequestParam String event_name, Model model, HttpSession session){

        ArrayList<DetailList> detailList = detailMapper.sellect(event_name);
        model.addAttribute("detailList". detailList);
        
        return "detail";
    }

    @GetMapping("/board")
    public String boardPage(HttpSession session){
        return "board";
    }

}
